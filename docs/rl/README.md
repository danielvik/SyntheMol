# SyntheMol-RL

Below are instruments for reproducing the results from the paper [SyntheMol-RL: a flexible reinforcement learning framework for designing novel and synthesizable antibiotics](https://www.biorxiv.org/content/10.1101/2025.05.17.654017v1), which uses the reinforcement learning (
RL) version of SyntheMol.

**Chemprop v2 note:** The instructions in this folder reference Chemprop v1 CLI commands. If you are using Chemprop
v2.2.2 (the default for this repo), replace `chemprop_train`/`chemprop_predict` with `chemprop train`/`chemprop predict`,
and convert any v1 checkpoints using `chemprop convert --conversion v1_to_v2`.

## Data

The relevant data should be downloaded to `SyntheMol/data`. This can be done as follows:

```bash
DATA_PATH=$(python -c "import synthemol; from pathlib import Path; print(Path(synthemol.__path__[0]).parent)")/rl
mkdir $DATA_PATH
gdown "https://drive.google.com/drive/folders/1Yex2UBjjmNFMZjBXA-ToVYMkpDFOorr0?usp=drive_link" --folder -O $DATA_PATH
```

Note that here, we use the 2022 q1-2 version of the building blocks and the 2022 q1-2 version of the enumerated REAL
Space molecules.

## Instructions

[real.md](real.md): Instructions for processing Enamine REAL building blocks, reactions, and molecules.

[wuxi.md](wuxi.md): Instructions for processing the WuXi building blocks, reactions, and molecules (May 2023 release).

[antibiotics.md](antibiotics.md): Instructions for generating antibiotic candidates for _Staphylococcus aureus_ using
reinforcement learning (RL). Includes instructions for processing antibiotics data, training antibacterial activity
prediction models, generating molecules with SyntheMol-RL, and selecting candidates.

[ablations.md](ablations.md): Instructions for reproducing the ablation experiments in the paper to determine the
importance of various components of SyntheMol-RL.

[gflownet.md](gflownet.md): Instructions for reproducing the GFlowNet experiments in the paper.
