# Computer Vision Project (CLIP-SEM ReID)

This repository trains a CLIP-based ReID model with optional SEM/AFEM modules and supports ablation studies on VeRi-776.

## Project structure

- configs/
	- default.yaml: main config (data, model, loss, optim, train, eval, output).
	- ablation/: ablation configs that override parts of default.yaml.
- data/
	- dataset.py: VeRi-776 dataset loading and dataset splits.
	- sampler.py: SpatioTemporalPKSampler for P x K sampling.
	- transforms.py: train and validation transforms.
- engine/
	- trainer.py: train_one_epoch loop and logging stats.
	- evaluator.py: evaluation logic and metrics (mAP, CMC).
	- re_ranking.py: re-ranking implementation used by evaluation.
- loss/
	- losses.py: LabelSmoothingCrossEntropy and SupConLoss.
- models/
	- backbone.py: ResNet-18 backbone.
	- sem.py: CLIP ViT-B/16 semantic encoder with resized positional embeddings.
	- afem.py: AFEM module for feature enhancement.
	- clip_senet.py: full model wiring (backbone + SEM + AFEM + heads).
- scripts/
	- train.py: training entrypoint (config load, loaders, model, optimizer).
	- run_ablation.sh: sweep all configs x seeds.
	- rerank_checkpoints.py: rerank evaluation for saved checkpoints.
- outputs/: experiment artifacts (checkpoints and logs).
- tests/: unit tests for core modules.
- requirements.txt: Python dependencies.
- pytest.ini: pytest configuration.

## Setup

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Data

The default dataset path is configured in `configs/default.yaml` under `data.root`:

```
/workspace/VeRi-776/VeRi
```

Update that path if your dataset lives elsewhere.
@
## Training

Run a single training job:

```bash
python scripts/train.py --config configs/default.yaml --seed 42 --output-dir outputs/default/seed_42
```

Logs are written to `outputs/<experiment>/seed_<seed>/logs` and checkpoints to `best.pth` and `last.pth`.

## Ablations

Run the full ablation sweep (all configs x seeds):

```bash
bash scripts/run_ablation.sh
```

Run a single ablation config and seed:

```bash
python scripts/train.py --config configs/ablation/abl_no_supcon.yaml --seed 2023 --output-dir outputs/abl_no_supcon/seed_2023
```

## Evaluation and rerank

Evaluation runs during training based on `train.eval_period`. You can toggle rerank behavior in the config under `eval.rerank`. For offline reranking of existing checkpoints, use `scripts/rerank_checkpoints.py`.

## Visualization

To run the full visualization suite (extracting t-SNE features, retrieval results for success/near miss/hard fail, and Grad-CAM heatmaps) across all ablation configs and standard seeds, use the provided bash script:

```bash
bash scripts/visualization.sh
```

You can also run visualization manually for any specific checkpoint. For example:

```bash
# 1. Extract t-SNE, Near Miss, Hard Fail
python scripts/extract_visuals.py --config configs/default.yaml --checkpoint outputs/default/seed_42/best.pth --outputs-dir outputs/visuals/seed_42

# 2. Generate Grad-CAM heatmaps
python scripts/generate_gradcam.py --config configs/default.yaml --checkpoint outputs/default/seed_42/best.pth --output-dir outputs/visuals/seed_42/gradcam
```

## Full Pipeline Demo

Below is an end-to-end example of installing dependencies, training a model with a custom seed (e.g., `777`, which is different from the default seeds `42`, `3407`, `2023`), reranking, and manually running the visualization tools for that specific seed.

**1. Install dependencies**
```bash
python -m pip install -r requirements.txt
```

**2. Train the model (Demo mode with 10 epochs)**
```bash
python scripts/train.py --config configs/default.yaml --seed 777 --epochs 10 --output-dir outputs/default/seed_777
```

**3. Rerank the saved checkpoint**
```bash
python scripts/rerank_checkpoints.py --outputs-root outputs/default/seed_777
```

**4. Extract Visualizations (t-SNE, Near Miss, Hard Fail)**
This will extract features for t-SNE and save grids of retrieval success, near miss, and hard fail cases.
```bash
python scripts/extract_visuals.py \
    --config configs/default.yaml \
    --checkpoint outputs/default/seed_777/best.pth \
    --outputs-dir outputs/visuals/seed_777
```

**5. Generate Grad-CAM Visualizations**
```bash
python scripts/generate_gradcam.py \
    --config configs/default.yaml \
    --checkpoint outputs/default/seed_777/best.pth \
    --output-dir outputs/visuals/seed_777/gradcam
```

## Tests

```bash
pytest -q
```
