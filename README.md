# Promptable Paleochannel Segmentation with SAM

A hackathon project for interactive segmentation of paleochannel bodies on RGB spectral decomposition maps using Segment Anything Models (SAM).

The project explores human-in-the-loop seismic interpretation, where a user guides the segmentation process with a small number of prompts, and the model acts as a general-purpose segmentation prior rather than a fully automatic interpreter.

---

## Project Goals

- demonstrate feasibility of SAM-based promptable segmentation for seismic attribute maps;
- build a working prototype within hackathon constraints;
- reduce manual paleochannel delineation time while preserving geological control;
- design a clean, extensible codebase suitable for further research or product development.

---

## Core Idea

RGB spectral decomposition maps are commonly used to visualize fluvial systems and paleochannels, but manual interpretation is time-consuming and subjective.

This project applies promptable segmentation models to:
- extract paleochannel geobodies with minimal user interaction;
- iteratively refine segmentation results;
- keep geological interpretation decisions under user control.

The model assists the interpreter — it does not replace geological reasoning.

---

## Key Features

- interactive segmentation using point and bounding-box prompts;
- support for RGB spectral decomposition maps (post-stack);
- fast iterative refinement of segmentation results;
- optional caching of image embeddings for interactive workflows;
- export of segmentation masks for downstream analysis or ML training;
- reproducible experiments and simple evaluation metrics.

---

## Repository Structure

```text
├── .github/
│   └── workflows/            # CI/CD configuration
│       ├── ci.yml
│       └── deploy.yml
│
├── docs/                     # Project documentation
│   ├── architecture.md       # High-level architecture
│   ├── requirements.txt      # Python dependencies (may be duplicated in root)
│   └── README.md             # Extended documentation
│
├── data/                     # Datasets
│   ├── raw/                  # Raw seismic RGB maps
│   ├── processed/            # Preprocessed data
│   └── annotations/          # Masks and prompt annotations
│
├── src/                      # Core source code
│   ├── preprocessing.py      # Data preprocessing
│   ├── segmentation.py       # SAM-based segmentation logic
│   ├── postprocessing.py     # Mask refinement and filtering
│   ├── evaluation.py         # Metrics and evaluation utilities
│   └── utils.py              # Shared helpers
│
├── experiments/              # Experiment outputs and logs
│   ├── experiment_1/
│   │   ├── logs.csv
│   │   └── results.json
│   └── experiment_2/
│
├── tests/                    # Unit and integration tests
│   ├── test_preprocessing.py
│   ├── test_segmentation.py
│   └── test_postprocessing.py
│
├── notebooks/                # Research and demo notebooks
│   ├── Exploratory_Data.ipynb
│   └── Train_Model.ipynb
│
├── requirements.txt          # Python dependencies
├── setup.py                  # Package setup (optional)
├── Makefile                  # Build and utility commands
├── LICENSE
└── .gitignore

чатгпт
