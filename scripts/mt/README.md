# Machine Translation
Unofficial implementation based on A. Vaswani et al., "Attention is all you need," in *NeurIPS*, 2017, pp. 5998-6008.

## Download dataset
```bash
./download.sh <download_dir>
```

## Preprocess dataset
```bash
poetry run python preprocess.py --data_dir <download_dir> --out_dir "data"
```

## Train Tokenizer
```bash
poetry run python tokenizer.py \
    --text_path "data/train.txt" \
    --out_dir "results" \
    --vocab_size 37000 \
    --model_type "bpe"
```

## Train model
```bash
poetry run python ../../src/aoarashi/tasks/mt/train.py \
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
poetry run python ../../src/aoarashi/tasks/mt/evaluate.py \
    --config-path="$(pwd)" \
    --config-name="config" \
    dataset.test_json_path="data/test.json" \
    tokenizer.model_path="results/tokenizer.model" \
    evaluate.out_dir="results" \
    evaluate.model_path="results/best_model.pt"
```
