"""
Collect metrics from all model results under results_yes/.
Produces metrics_summary.json with anchor results first, then absolute,
for each model. own_timexer is further split by single_endo / twoendo.
"""

import json
from pathlib import Path

BASE = Path(__file__).parent / "results_yes"
BATCHES = {"KAU084", "KAU081", "KAU071", "KAU079"}
BATCH_ORDER = ["KAU084", "KAU081", "KAU071", "KAU079"]
MODELS = ["csdi", "itransformer", "mambats", "ncde", "own_timexer", "patchtst", "tft", "timeXer", "timellm"]


def load_metrics(path: Path) -> dict:
    with open(path) as f:
        data = json.load(f)
    entry = {}
    for key in ("metrics", "metrics_spike", "metrics_flat"):
        if key in data:
            entry[key] = data[key]
    return entry


def collect_under(root: Path, name_aliases: dict = None, exclude: set = None) -> dict:
    """
    Recursively collect results.json files under root.
    Returns a nested dict:
      - If root contains batch folders directly: { "KAU084": {...}, ... }
      - If there is one extra layer of named subfolders:
        { "subfolder_name": { "KAU084": {...}, ... }, ... }
    Only exact BATCHES names are included (plus any entries in name_aliases).
    name_aliases maps a specific folder name to its canonical batch key,
    e.g. {"KAU084_revin_all_var": "KAU084"}.
    """
    if not root.exists():
        return {}

    aliases = name_aliases or {}
    excluded = exclude or set()

    def batch_key(name: str) -> str | None:
        if name in BATCHES:
            return name
        if name in aliases:
            return aliases[name]
        return None

    # Check direct batch children first
    direct = {}
    subdirs = {}
    for child in sorted(root.iterdir()):
        if not child.is_dir() or child.name in excluded:
            continue
        key = batch_key(child.name)
        if key is not None:
            rj = child / "results.json"
            if rj.exists():
                direct[key] = load_metrics(rj)
        else:
            # Could be an intermediate named folder
            subdirs[child.name] = child

    if direct:
        return {b: direct[b] for b in BATCH_ORDER if b in direct}

    # One extra layer: named experiment subfolder → batch folders
    result = {}
    for name, subdir in subdirs.items():
        batches = {}
        for child in sorted(subdir.iterdir()):
            if not child.is_dir():
                continue
            key = batch_key(child.name)
            if key is not None:
                rj = child / "results.json"
                if rj.exists():
                    batches[key] = load_metrics(rj)
        if batches:
            result[name] = {b: batches[b] for b in BATCH_ORDER if b in batches}
    return result

def collect_timellm(model_dir: Path) -> dict:
    """timellm uses anchor / absolute instead of anchor/absolute."""
    result = {"anchor": {}, "absolute": {}}
    for child in sorted(model_dir.iterdir()):
        if not child.is_dir():
            continue
        name_lower = child.name.lower()
        if "anchor" in name_lower:
            key = "anchor"
        elif "absolute" in name_lower:
            key = "absolute"
        else:
            continue
        batches = {}
        for batch_dir in sorted(child.iterdir()):
            if batch_dir.is_dir() and batch_dir.name in BATCHES:
                rj = batch_dir / "results.json"
                if rj.exists():
                    batches[batch_dir.name] = load_metrics(rj)
        if batches:
            result[key][child.name] = {b: batches[b] for b in BATCH_ORDER if b in batches}
    return result
    
def collect_patchtst(model_dir: Path) -> dict:
    """patchtst uses reactor_patchtst_anchor / reactor_patchtst_absolute instead of anchor/absolute."""
    result = {"anchor": {}, "absolute": {}}
    for child in sorted(model_dir.iterdir()):
        if not child.is_dir():
            continue
        name_lower = child.name.lower()
        if "anchor" in name_lower:
            key = "anchor"
        elif "absolute" in name_lower:
            key = "absolute"
        else:
            continue
        batches = {}
        for batch_dir in sorted(child.iterdir()):
            if batch_dir.is_dir() and batch_dir.name in BATCHES:
                rj = batch_dir / "results.json"
                if rj.exists():
                    batches[batch_dir.name] = load_metrics(rj)
        if batches:
            result[key][child.name] = {b: batches[b] for b in BATCH_ORDER if b in batches}
    return result


def collect_own_timexer(model_dir: Path) -> dict:
    """own_timexer: anchor/absolute → single_endo/twoendo → batch."""
    result = {"anchor": {}, "absolute": {}}
    for mode in ("anchor", "absolute"):
        mode_dir = model_dir / mode
        if not mode_dir.exists():
            continue
        for modality_dir in sorted(mode_dir.iterdir()):
            if not modality_dir.is_dir():
                continue
            modality = modality_dir.name
            batches = {}
            for batch_dir in sorted(modality_dir.iterdir()):
                if batch_dir.is_dir() and batch_dir.name in BATCHES:
                    rj = batch_dir / "results.json"
                    if rj.exists():
                        batches[batch_dir.name] = load_metrics(rj)
            if batches:
                result[mode][modality] = {b: batches[b] for b in BATCH_ORDER if b in batches}
    return result


def main():
    summary = {}

    for model in MODELS:
        model_dir = BASE / model
        if not model_dir.exists():
            print(f"  WARNING: {model} not found, skipping")
            continue

        if model == "own_timexer":
            summary[model] = collect_own_timexer(model_dir)
        elif model == "patchtst":
            summary[model] = collect_patchtst(model_dir)
        elif model == "timellm":
            summary[model] = collect_timellm(model_dir)
        else:
            anchor = collect_under(model_dir / "anchor")
            aliases = {"KAU084_revin_all_var": "KAU084"} if model == "csdi" else {}
            excl = {"reactor_itransformer_conservative_noresidual"} if model == "itransformer" else set()
            absolute = collect_under(model_dir / "absolute", name_aliases=aliases, exclude=excl)
            summary[model] = {"anchor": anchor, "absolute": absolute}

    out_path = Path(__file__).parent / "metrics_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Written: {out_path}")

    # Print a quick count summary
    for model, data in summary.items():
        if model == "own_timexer":
            for mode in ("anchor", "absolute"):
                for modality, batches in data[mode].items():
                    print(f"  {model}/{mode}/{modality}: {list(batches.keys())}")
        else:
            for mode in ("anchor", "absolute"):
                entries = data.get(mode, {})
                if not entries:
                    continue
                # Could be flat {KAU...: ...} or nested {exp: {KAU...: ...}}
                sample = next(iter(entries.values()))
                if isinstance(sample, dict) and "metrics" in sample:
                    # Flat
                    print(f"  {model}/{mode}: {list(entries.keys())}")
                else:
                    # Nested
                    for exp, batches in entries.items():
                        print(f"  {model}/{mode}/{exp}: {list(batches.keys())}")


if __name__ == "__main__":
    main()
