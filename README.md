# SNR_Optimization

This repository contains a pipeline to measure and compare amplitude-based signal-to-noise ratio (SNR) in both sensor space and source space for sensory evoked potentials.

All dependencies and the expected folder structure are described in the notebook: SNR_Optimization.ipynb.

The dataset accompanying this pipeline is available on Zenodo:
DOI: 10.5281/zenodo.19063904

After downloading and extracting the dataset, the following folders must be placed in the root directory of this repository:

SNR_Optimization/
│
├── subject_1/
├── processed_data/
├── SNR_Optimization.ipynb
└── functions_01

Folder description

subject_1/
Contains the head model used in this study.

processed_data/
Contains all participant data, including:

eeg/ → preprocessed EEG epochs

leadfield/ → forward models

current/ → reconstructed source activity and SNR analysis

Each participant has their data organized within this structure.

# Usage

Ensure the dataset folders are correctly placed in the root directory.

Open and run SNR_Optimization.ipynb.

# Notes

The preprocessed EEG epochs for all participants are located in:
processed_data/<participant>/eeg/
The pipeline consists of two main parts:
1. ERP and Source locaization computation 
2. SNR analysis (sensor and source space) and figure generation

If the EEG and current data are already available, you can skip the first part and directly run the figure generation section.

# Reproducibility

This repository is designed to work directly with the provided dataset.
Please ensure the folder structure is preserved exactly as described above.


