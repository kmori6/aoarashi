# Automatic Speech Recognition
Unofficial implementation based on A. Gulati et al., "Conformer: Convolution-augmented Transformer for Speech Recognition," in *Interspeech*, 2020, pp. 5036-5040.

## Download dataset
```bash
./download.sh <download_dir>
```

## Preprocess dataset
```bash
poetry run python preprocess.py \
    --data_dir <download_dir> \
    --out_dir "data" \
    --min_duration 1.0 \
    --max_duration 20.0
```

## Train Tokenizer
```bash
# NOTE: true vocab_size is 999 + 1 = 1000 where we reserve the last toke id for blank
poetry run python ../mt/tokenizer.py \
    --text_path "data/train.txt" \
    --out_dir "results" \
    --vocab_size 999 \
    --model_type "bpe"
```

## Train model
```bash
poetry run python -m konpeki.tasks.asr.train \
    --config-path=$(pwd) \
    --config-name="config" \
    dataset.train_json_path="data/train.json" \
    dataset.valid_json_path="data/valid.json" \
    tokenizer.model_path="results/tokenizer.model" \
    train.out_dir="results" \
    train.checkpoint_path=
```

## Evaluate model
```bash
poetry run python -m konpeki.tasks.asr.evaluate \
    --config-path="$(pwd)" \
    --config-name="config" \
    dataset.test_json_path="data/test-clean.json" \
    tokenizer.model_path="results/tokenizer.model" \
    evaluate.out_dir="results" \
    evaluate.model_path="results/best_model.pt"
```
