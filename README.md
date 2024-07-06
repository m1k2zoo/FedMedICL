# FedMedICL
This is the official repository for the paper titled "FedMedICL: Towards Holistic Evaluation of Distribution Shifts in Federated Medical Imaging" accepted at MICCAI24. FedMedICL is a unified framework designed to address federated medical imaging challenges across various dimensions including label, demographic, and temporal distribution shifts. Unlike traditional benchmarks that assess performance under a single type of distribution shift, FedMedICL offers a more holistic evaluation across dynamic healthcare settings.


## Requirements
This project requires Python 3.8 or higher and PyTorch 1.8 or later. Follow these steps to set up the Conda environment:
```bash
conda create -n fedmedicl1 python=3.9.19
conda activate fedmedicl1
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu116
```

## Installation

### Cloning the Repository
To get started, clone our repository:
```bash
git clone https://github.com/m1k2zoo/FedMedICL.git
cd FedMedICL
```

## Datasets
For complete details on downloading and preprocessing the datasets, please refer to the [datasets documentation](docs/datasets.md).

## Quick Start
To get started with running experiments on our FedMedICL framework, we provide example scripts for each dataset using the FedAvg baseline located in the `scripts` directory:
- For main experiments, the scripts are named `FedAvg_<dataset>.sh`, where \<dataset\> is the dataset name.
- For novel disease outbreaks, use the script `novel_FedAvg_CheXCOVID.sh`.

Below is an example script for training with the HAM10000 dataset using the FedAvg baseline:
```bash
bash scripts/FedAvg_HAM10000.sh
```
The training results will be logged in the specified `output_dir` directory. 

### (Advanced) Additional Experiments
For advanced usage scenarios, such as experiments involving multiple datasets across clients, we provide an example under `scripts/additional_examples`.


## Re-producing our Experiments
The hyperparameters used in the FedAvg_<dataset>.sh and novel_FedAvg_CheXCOVID.sh scripts match those we used in our experiments described in the paper. All main experiment datasets share the same hyperparameters, except for CheXpert, which has slightly different ones to accommodate its large-scale size.

We also provide the Weights & Biases sweeps used to re-run our experiments, which include the hyperparameters used. These can be found in the `scripts/sweeps` directory.


## Supported Algorithms
For information on supported algorithms, please refer to the [supported algorithms documentation](docs/supported_algorithms.md).


## Reporting and Visualization
For details on how results are stored and organized, and for instructions on generating plots, please refer to the [reporting documentation](docs/reporting.md).



## Contact:

* **Kumail Alhamoud:**  kumail.hamoud@kaust.edu.sa
* **Yasir Ghunaim:** yasir.ghunaim@kaust.edu.sa
* **Motasem Alfarra:** motasem.alfarra@kaust.edu.sa
