"""
Collect per-batch metrics across training modalities for the 4 transfer-learning models:
  own_timexer, patchtst, itransformer, timellm

Modalities collected:
  transfer_learning  — results_yes/<model>/absolute single_endo (own_timexer) or direct absolute
  one_by_one         — results/one_by_one/<model>/reactor_*/<batch>/results.json
  concatenated       — results/concatenated_test/<model>/no_token*/results.json
  loo                — results/loo/<model>/LOO_<batch>*/results.json
  finetune           — results/finetune/<model>/(reactor_*/)?test*/finetune_*/results.json

Produces modalities_summary.json with structure:
  { modality: { model: { batch: { metrics, metrics_spike, metrics_flat } } } }
"""

import json
from pathlib import Path

BASE_YES  = Path(__file__).parent / "results_yes"
BASE_RES  = Path(__file__).parent / "results"

MODELS    = ["own_timexer", "patchtst", "itransformer", "timellm"]
BATCH_ORDER = ["KAU084", "KAU081", "KAU071", "KAU079"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_old_format(path: Path) -> dict | None:
    """Old results.json: metrics/metrics_spike/metrics_flat at top level."""
    if not path.exists():
        return None
    with open(path) as f:
        d = json.load(f)
    entry = {}
    for key in ("metrics", "metrics_spike", "metrics_flat"):
        if key in d:
            entry[key] = d[key]
    return entry if entry else None


def per_exp_to_entry(per_exp: dict, batch: str) -> dict | None:
    """Convert per_experiment_metrics[batch] (overall/spike/non_spike) to standard entry."""
    b = per_exp.get(batch)
    if not b:
        return None
    def pick(section):
        raw = b.get(section, {})
        # flatten variable level: take first variable's values
        for var_vals in raw.values():
            return var_vals
        return {}
    return {
        "metrics":       {"nh4": pick("overall")},
        "metrics_spike": {"nh4": pick("spike")},
        "metrics_flat":  {"nh4": pick("non_spike")},
    }


def load_new_format(path: Path, batch: str) -> dict | None:
    """New results.json: per_experiment_metrics[batch]{overall/spike/non_spike}."""
    if not path.exists():
        return None
    with open(path) as f:
        d = json.load(f)
    return per_exp_to_entry(d.get("per_experiment_metrics", {}), batch)


def load_new_all_batches(path: Path) -> dict:
    """Read all batches from per_experiment_metrics in a single results.json."""
    if not path.exists():
        return {}
    with open(path) as f:
        d = json.load(f)
    per_exp = d.get("per_experiment_metrics", {})
    result = {}
    for batch, data in per_exp.items():
        entry = per_exp_to_entry(per_exp, batch)
        if entry:
            result[batch] = entry
    return result


# ---------------------------------------------------------------------------
# 1. Transfer learning (results_yes, absolute)
# ---------------------------------------------------------------------------

def collect_transfer_learning() -> dict:
    """
    Per-model search roots in results_yes/absolute:
      own_timexer  → absolute/single_endo/<batch>/
      patchtst     → reactor_patchtst_absolute/<batch>/   (no absolute/ level)
      itransformer → absolute/reactor_itransformer_noresidual/<batch>/
      timellm      → absolute/<batch>/
    """
    TRANSFER_ROOTS = {
        "own_timexer":  BASE_YES / "own_timexer"  / "absolute" / "single_endo",
        "patchtst":     BASE_YES / "patchtst"      / "reactor_patchtst_absolute",
        "itransformer": BASE_YES / "itransformer"  / "absolute" / "reactor_itransformer_noresidual",
        "timellm":      BASE_YES / "timellm"       / "absolute",
    }
    data = {}
    for model in MODELS:
        root = TRANSFER_ROOTS.get(model)
        if not root or not root.exists():
            continue
        batches = {}
        for child in root.iterdir():
            if child.is_dir() and child.name in BATCH_ORDER:
                entry = load_old_format(child / "results.json")
                if entry:
                    batches[child.name] = entry
        if batches:
            data[model] = {b: batches[b] for b in BATCH_ORDER if b in batches}
    return data


# ---------------------------------------------------------------------------
# 2. One-by-one (results/one_by_one)
# ---------------------------------------------------------------------------

def collect_one_by_one() -> dict:
    data = {}
    base = BASE_RES / "one_by_one"
    for model in MODELS:
        model_dir = base / model
        if not model_dir.exists():
            continue
        batches = {}
        # model/reactor_*/KAU*/results.json
        for child in model_dir.iterdir():
            if not child.is_dir():
                continue
            for batch_dir in child.iterdir():
                if batch_dir.is_dir() and batch_dir.name in BATCH_ORDER:
                    entry = load_new_format(batch_dir / "results.json", batch_dir.name)
                    if entry:
                        batches[batch_dir.name] = entry
        if batches:
            data[model] = {b: batches[b] for b in BATCH_ORDER if b in batches}
    return data


# ---------------------------------------------------------------------------
# 3. Concatenated (results/concatenated_test)
# ---------------------------------------------------------------------------

def collect_concatenated() -> dict:
    data = {}
    base = BASE_RES / "concatenated_test"
    for model in MODELS:
        model_dir = base / model
        if not model_dir.exists():
            continue

        # timellm: results.json directly in the model folder
        if model == "timellm":
            rj = model_dir / "results.json"
            batches = load_new_all_batches(rj)
        else:
            # Find subfolder whose name starts with no_token or notoken (case-insensitive)
            target = None
            for child in model_dir.iterdir():
                if child.is_dir() and child.name.lower().replace("_", "").startswith("notoken"):
                    target = child
                    break
            if target is None:
                continue
            batches = load_new_all_batches(target / "results.json")

        if batches:
            data[model] = {b: batches[b] for b in BATCH_ORDER if b in batches}
    return data


# ---------------------------------------------------------------------------
# 4. Leave-one-out (results/loo)
# ---------------------------------------------------------------------------

def collect_loo() -> dict:
    data = {}
    base = BASE_RES / "loo"
    for model in MODELS:
        model_dir = base / model
        if not model_dir.exists():
            continue
        batches = {}
        # Only direct LOO_KAU* folders (not nested subfolders like reactor_itransformer)
        for child in model_dir.iterdir():
            if not child.is_dir() or not child.name.startswith("LOO_"):
                continue
            rj = child / "results.json"
            if not rj.exists():
                continue
            with open(rj) as f:
                d = json.load(f)
            # Identify the held-out batch from per_experiment_metrics key
            per_exp = d.get("per_experiment_metrics", {})
            if not per_exp:
                continue
            batch = next(iter(per_exp))
            if batch not in BATCH_ORDER:
                continue
            # Use top-level metrics (not per_experiment_metrics)
            entry = load_old_format(rj)
            if entry:
                batches[batch] = entry
        if batches:
            data[model] = {b: batches[b] for b in BATCH_ORDER if b in batches}
    return data


# ---------------------------------------------------------------------------
# 5. Finetune (results/finetune)
# ---------------------------------------------------------------------------

def collect_finetune() -> dict:
    data = {}
    base = BASE_RES / "finetune"
    for model in MODELS:
        model_dir = base / model
        if not model_dir.exists():
            continue
        batches = {}

        def _scan_tests(tests_root: Path):
            for test_dir in tests_root.iterdir():
                if not test_dir.is_dir() or not test_dir.name.startswith("test"):
                    continue
                for sub in test_dir.iterdir():
                    if sub.is_dir() and sub.name.startswith("finetune_"):
                        rj = sub / "results.json"
                        if not rj.exists():
                            continue
                        with open(rj) as f:
                            d = json.load(f)
                        per_exp = d.get("per_experiment_metrics", {})
                        for batch, _ in per_exp.items():
                            entry = per_exp_to_entry(per_exp, batch)
                            if entry and batch in BATCH_ORDER:
                                batches[batch] = entry

        if model == "timellm":
            # No reactor_ layer
            _scan_tests(model_dir)
        else:
            # model/reactor_something/test*/finetune_*/results.json
            # Skip old_scaler_bug sibling
            for child in model_dir.iterdir():
                if child.is_dir() and child.name.startswith("reactor_"):
                    _scan_tests(child)

        if batches:
            data[model] = {b: batches[b] for b in BATCH_ORDER if b in batches}
    return data


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    summary = {
        "transfer_learning": collect_transfer_learning(),
        "one_by_one":        collect_one_by_one(),
        "concatenated":      collect_concatenated(),
        "loo":               collect_loo(),
        "finetune":          collect_finetune(),
    }

    out_path = Path(__file__).parent / "modalities_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Written: {out_path}")

    for modality, models in summary.items():
        for model, batches in models.items():
            print(f"  {modality}/{model}: {list(batches.keys())}")


if __name__ == "__main__":
    main()
