# MProto

## Usage

### Environment setup and preparation

```bash
conda create --name mproto --file requirements.txt
```

You should first download pre-trained language model (from huggingface) and
set `plm_path` and `tokenizer_path` in the configuration file to the path of the pre-trained language model.

We use bert-base-cased for CoNLL03 dataset and biobert-base-cased-v1.1 for BC5CDR datset.

### Train

We train our model using RTX-3090 on all datasets.

Type the following command to train and test the model.

```bash
python -m scripts.train_and_test cfg/conll03-dict/mproto/train-p3-1.0.toml
```
