## Pytorch implementation of [Neural Sequence Decoder](https://github.com/fwillett/speechBCI/tree/main/NeuralDecoder)

## Requirements
- python >= 3.9
- pytorch >= 1.9.0
- CUDA-capable GPU (recommended)

## Installation

```bash
pip install -e .
```

## Dataset Preparation

1. Convert the speech BCI dataset using [formatCompetitionData.ipynb](./notebooks/formatCompetitionData.ipynb)
2. The formatted dataset should be saved as `data/formatted/ptDecoder_ctc`

## Training

Train a new model:
```bash
python ./scripts/train_model.py
```

## Using the Best Model (run16_5)

The best performing model is **run16_5**, which achieved the lowest phoneme error rate (PER) on the test set. Model checkpoints and training metadata are stored in `data/checkpoints/run16_5/`.

**Note:** Model weight files (`*.pt` and `modelWeights`) are not included in this repository due to their large size (~517MB each). To use the model, you'll need to either:
1. Train the model yourself using the configuration in `data/checkpoints/run16_5/train.log`
2. Download the model weights separately if available

### Evaluating the Model

The evaluation script requires:
- `args` file (included in repo) - contains model architecture and training hyperparameters
- Model weights file - either `modelWeights` or `best_model.pt` (not in repo due to size)

If you have the model weights, you can evaluate the model using:

```bash
python scripts/eval.py \
    --modelPath data/checkpoints/run16_5 \
    --datasetPath data/formatted/ptDecoder_ctc \
    --split test \
    --out eval_results.json
```

The script will automatically detect and load the model weights from `modelWeights`, `best_model.pt`, or other checkpoint files in the model directory.

### Available Files in Repository

For each training run in `data/checkpoints/`, the following files are included:
- `args` - Training hyperparameters (pickle file)
- `train.log` - Training log with configuration details
- `metrics.jsonl` - Step-by-step training metrics (loss, PER, learning rate, etc.)
- `trainingStats` - Per-epoch training statistics (pickle file)
- `eval*.json` - Evaluation results on test/competition sets

Model weights (`*.pt` files and `modelWeights`) are excluded due to size constraints.

