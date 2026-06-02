# Unibio Thesis Code Repository

This repository contains code developed as part of my MSc thesis project at Aalborg University, carried out in collaboration with Unibio A/S.

The thesis investigated probabilistic time-series forecasting models for soft-sensor development in a bioreactor context, with a focus on nitrogen-related process variables and short-term concentration forecasting.

## Purpose of this repository

The repository is intended to document the code structure, model implementations, data treatment logic, and experimental workflows used during the thesis.

It is provided for academic transparency and reproducibility of the methodological approach. However, the original industrial dataset is not included.

## Confidentiality statement

This repository does **not** contain confidential Unibio data.

Specifically, the repository does not include:

- raw industrial data;
- processed industrial datasets;
- internal company documents;
- credentials, passwords, tokens, or private access information;
- trained model checkpoints based on confidential data;
- proprietary Unibio files or internal reports.

The code may contain references to expected data formats, variable names, or local file paths used during development, but the actual data files are intentionally omitted due to confidentiality restrictions.

## Repository structure

```text
unibio_thesis/
│
├── Batch_processing_and_models/
│   ├── eda/
│   ├── metrics_collector_export/
│   ├── csdi_train_reactor_probabilistic.py
│   ├── itransformer_train_reactor_probabilistic.py
│   ├── mambats_train_reactor_probabilistic.py
│   ├── modified_timexer_train_reactor_probabilistic.py
│   ├── ncde_train_reactor_probabilistic.py
│   ├── patchtst_train_reactor_probabilistic.py
│   ├── tft_train_reactor_probabilistic.py
│   ├── timellm_train_reactor_probabilistic.py
│   └── timexer_train_reactor_probabilistic.py
│
├── WWTP/
│   ├── plot_horizon.py
│   ├── train_nh4_endo_only.py
│   ├── train_nh4_exo_o2.py
│   ├── train_nh4_gaussian.py
│   ├── vanilla_nh4_exo_o2.py
│   └── vanilla_timexer.py
│
└── README.md
