# MLIII Group 3: Facial Age and Sex Prediction

We are comparing age-only, sex-only, and joint age-and-sex VGG16 models, then testing whether rotation, random erasing, or both improve performance.

## What is finished

The no-augmentation baselines ran for 10 epochs across seeds 101, 202, and 303. All models used the same finalized images and person-disjoint splits. Checkpoints were selected using validation loss, then evaluated on the held-out test set.

| Model | Mean test age MAE | Mean test sex accuracy |
|---|---:|---:|
| Age-only | 6.6143 years | — |
| Sex-only | — | 95.8330% |
| Joint | 6.7892 years | 95.5705% |

Joint learning slightly worsened both tasks in all three matched seeds under this recipe.

## Files to review

- `MLIII_Group3_Project_Code.ipynb`: annotated baseline implementation.
- `results/baselines/test_summary.csv`: means and sample standard deviations.
- `results/baselines/test_results_per_seed.csv`: individual test results.
- `results/baselines/test_paired_contrasts.csv`: joint-versus-single-task differences.
- `results/baselines/protocol.json`: baseline comparison settings.
- `data/cohort/`: exact split assignments, cohort protocol, and source checksums.

Images, archives, checkpoints, and generated experiment folders are not included.

## Before running the notebook

Use an HPC GPU session and the project's Python environment. The notebook needs PyTorch, torchvision, NumPy, pandas, SciPy, scikit-learn, Pillow, matplotlib, and optionally tqdm.

Start the notebook kernel in the repository root. The code uses `Path.cwd()` for its working directory.

Restore the finalized split files to the location expected by the notebook:

```bash
mkdir -p studies/vgg16_common_cohort_v1
cp data/cohort/* studies/vgg16_common_cohort_v1/
```

The working directory also needs access to:

- `imdb_crop.tar`
- `imdb_meta.tar`
- `imdb_crop/` and `imdb/imdb.mat`, if already extracted
- The matching `dataset_extraction.json`, if reusing a completed extraction

Use your own dataset copy or an authorized shared location. Do not assume another teammate's scratch directory is accessible. Agree on dataset access before launching jobs so we avoid unnecessarily repeating extraction.

## Execution flags

Leave these settings off when reviewing:

```python
RUN_EXPERIMENTS = False
EVALUATE_TEST = False
```

With execution enabled, the current notebook trains nine fresh NO-AUGMENTATION runs and then evaluates them on test data. It does not yet implement the augmentation experiments.

Running the settings cell creates a new timestamped session folder even with execution disabled. To review finished baselines, open the CSVs in `results/baselines/`; Run All does not automatically reload those results.

Save the notebook before launching training so each run captures the saved code. Ensure the HPC allocation has enough remaining time.

## Next experiments

Implement rotation, random erasing, and their combination using the settings in the saved cohort protocol. Keep the cohort, splits, seeds, architecture, optimizer, training budget, and loss weights fixed.

Apply random augmentation only during training. Keep validation and test preprocessing deterministic. Give each condition clearly labeled settings and output folders.

Review the implementation before launching the 27 additional training runs. Keep checkpoint selection validation-based and do not adjust settings based on test outcomes.

## AI assistance

I used OpenAI's ChatGPT (Codex) to help plan the code structure and experimental workflow, explain PyTorch concepts, and generate and revise portions of the notebook. I reviewed and adapted the code and take responsibility for its use, the experiments, and the interpretation of results.

## Creating a Python environment

The baseline runs used Python 3.12.14. Create your own environment rather than changing another teammate's environment.

From the repository root:

```bash
conda create -n mliii-group3 python=3.12.14 -y
conda activate mliii-group3

python -m pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements.txt

python -m ipykernel install --user --name mliii-group3 --display-name "MLIII Group 3"
```

In Code Server, select **MLIII Group 3** as the notebook kernel.

Check GPU availability from a notebook cell inside your allocated GPU session:

```python
import torch
print(torch.__version__)
print(torch.cuda.is_available())
```

The expected PyTorch version is `2.7.0+cu126`. GPU availability should be `True` inside a correctly configured GPU allocation.

The original environment was named `ds6050`; teammates do not need access to my personal environment or use that same name.
