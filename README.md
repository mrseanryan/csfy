# classifier trainer

Train a simple and fast text classifier, based on [DistilBERT](https://huggingface.co/distilbert/distilbert-base-uncased).

- much faster and lighter than an LLM
- fully configurable, including the base model
- can train on data that has multiple custom labels

Key points about the base model:

> DistilBERT is a transformers model, smaller and faster than BERT
> This model is uncased: it does not make a difference between english and English.
> this model is primarily aimed at being fine-tuned on tasks that use the whole sentence (potentially masked) to make decisions, such as sequence classification, token classification or question answering.

Key points about BERT (the base model of DistilBERT):

> Pretrained model on English language using a masked language modeling (MLM) objective
> pretrained on the raw texts only, with no humans labeling them in any way (which is why it can use lots of publicly available data)
> the model learns an inner representation of the English language that can then be used to extract features useful for downstream tasks: if you have a dataset of labeled sentences, for instance, you can train a standard classifier using the features produced by the BERT model as inputs

# Setup

1. Install [poetry](https://python-poetry.org/docs#installation)

2. Use poetry

```shell
poetry install
```

# Usage

To see the built-in help:

```
poetry run csfy
```

1. Prepare the dataset
2. Edit `config.ini` to suit your environment
- The dataset should be in parquet format and have 2 columns as set in config.ini: `COLUMN_TEXT` and `COLUMN_LABEL`
3. Train
```shell
poetry csfy train <path to input.parquet>
```
4. Test (Predict)
```shell
poetry csfy predict <path to model> <text>
```

Chat mode: (interactive loop)

```shell
poetry csfy predict <path to model> <text> --chat
```

## Optional ONNX conversion [optimization][language neutral]

1. Export to ONNX format

```shell
poetry export <path to model from 'train'> <path to ONNX model to export>
```

2. Reduce model size whilst maintaining most of the accuracy

```shell
poetry quantize <path to ONNX model> <path to output ONNX model> <quantization level>
```

3. Test (Predict) from ONNX model

```shell
poetry csfy predict <path to ONNX model> <text>
```

# Troubleshooting

- [GPU card not being used](./README.gpu.md)

- `poetry run csfy` does not list any commands
  - try running `poetry install` again or `poetry lock`

# Example datasets

## Natural language

### Clean

[alpaca](https://huggingface.co/datasets/tatsu-lab/alpaca)

### Toxic

[harmful_behaviors](https://huggingface.co/datasets/mlabonne/harmful_behaviors)

### Other

[kaggle](https://www.kaggle.com/search?q=datasets)
