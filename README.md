# VeRe: Verification Guided Fault Localization and Repair Synthesis of Deep Neural Networks

This repository contains the codes and scripts for the paper "**VeRe: Verification Guided Fault Localization and Repair Synthesis of Deep
Neural Networks**", which is a significant extension of our previous work published in ICSE 2024. It includes implementations for three main tasks: Backdoor Removal, Unfairness Mitigation, and Safety Property Correction.

## Requirements

To run the code, please ensure the following dependencies are installed:

- Python 3.9.19
- PyTorch 2.3.1
- auto_LiRPA: You can install auto_LiRPA from [here](https://github.com/Verified-Intelligence/auto_LiRPA).

## Models

Due to capacity limitations, we uploaded all the buggy models involved in the experiments of backdoor repair to this [anonymous link](https://drive.google.com/file/d/1_xlhfBSv99TWaLcvagdrFEGvPUlyH3Ku/view?usp=drive_link).
Other models are provided in `fairness/buggy_model` and `safety/buggy_model`.

## Datasets

For the **safety property correction** and **unfairness mitigation** tasks, the datasets are provided in their corresponding directories. 

For the **backdoor removal** task, due to the large size of the datasets, please download them manually. After downloading, you need to update the dataset storage path in the `data_model` function within `utils/prepare.py`.

## Structure
```text
.
├── backdoor/
│   ├── buggy_model/     models for backdoor experiment
│   ├── results/         logs and results for backdoor repair
│   ├── utils/           utility functions
│   ├── main.py          main entry for backdoor repair
│   ├── run_main.sh      script to reproduce main experiments
│   ├── run_arch.sh      script for reproduce experiments of repairing other activation functions
│   └── run_num.sh       script for reproduce experiments of repairing with different number of data
├── fairness/
│   ├── buggy_model/     models for fairness experiment
│   ├── data/            dataset for fairness task
│   ├── preprocessing/   code for data preprocessing
│   ├── main.py          main entry for fairness repair
│   └── run.sh           script to reproduce fairness experiments
└── safety/
    ├── model/           models for safety verification
    ├── results/         logs and results for safety repair
    ├── main.py          main entry for safety repair
    └── run.sh           script to reproduce safety experiments
```


## Reproducing the Experiments

We prepare scripts to quickly reproduce our experiments.
The output logs will show the Repair Success Rate (RSR), the generalization and the Accuracy (or drawdown rate).

### 1. Reproduce Repairing Backdoor

Navigate to the `backdoor` directory. We provide three scripts to reproduce different experimental settings. The results will be saved in `backdoor/results/`.

**Main Repair Experiment:**
Run the following command to repair backdoor models (Badnets and Blend attacks) on the CIFAR-10 dataset using 1000 repair samples:
```bash
cd backdoor
bash run_main.sh cifar
```

**Varying Number of Repair Data:**
Run the following command to automate the repair process while varying the number of available repair samples from 100 to 1000:
```bash
bash run_num.sh cifar
```

**Different Activation Functions:**
Run the following command to repair networks with different activation functions (e.g., LeakyReLU, GELU):
```bash
bash run_arch.sh cifar leakyrelu
# or
bash run_arch.sh cifar gelu
```

---

### 2. Reproduce Repairing Fairness

Navigate to the `fairness` directory. We provide a one-click script to automatically execute repair across all datasets and sensitive attributes. The results will be saved in `fairness/results/`.

Run the script with the desired number of repair samples (e.g., 200 samples for each group):
```bash
cd fairness
bash run.sh 200 200
```
*Note: This will automatically perform all steps including data preprocessing and model repair.*

---

### 3. Reproduce Repairing Safety Properties

Navigate to the `safety` directory. We provide a script to automatically reproduce the repair on all 36 buggy models. The results will be saved in `safety/results/`.

Run the script by specifying the number of counterexamples and normal samples (e.g., 500 counterexamples and 500 normal samples):
```bash
cd safety
bash run.sh 500 500
```

