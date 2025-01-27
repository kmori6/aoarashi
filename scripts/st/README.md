# Speech Translation
Unofficial implementation based on J. Xue et al., "Large-scale streaming end-to-end speech translation with neural transducers," in *Interspeech*, 2022, pp. 3263-3267.

## Download dataset
Download source language audio files and transcripts from [Common Voice Corpus 4](https://commonvoice.mozilla.org/en/datasets) mannually.

We suppose the source language is English and target language is Japanese.

The download file is named to *en.tar.gz* and has been already extracted in its directory by such as `tar -xvf en.tar.gz` here.
```bash
./download.sh <covost2_download_dir>
```

## Preprocess dataset
```bash
poetry run python preprocess.py --commonvoice_data_dir <commonvoice_download_dir> --covost2_data_dir <covost2_download_dir> --out_dir data
```

## Train Tokenizer
```bash
# NOTE: true vocab_size is 4095 + 1 = 4096 where we reserve the last toke id for blank
poetry run python ../mt/tokenizer.py \
    --text_path "data/train.txt" \
    --out_dir "results" \
    --vocab_size 4095 \
    --model_type "bpe"
```

## Train model
```bash
poetry run python -m konpeki.tasks.st.train \
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
poetry run python -m konpeki.tasks.st.evaluate \
    --config-path="$(pwd)" \
    --config-name="config" \
    dataset.test_json_path="data/test.json" \
    tokenizer.model_path="results/tokenizer.model" \
    evaluate.out_dir="results" \
    evaluate.model_path="results/best_model.pt"
```
