# Speech Separation
Unofficial implementation based on Y. Luo et al., "Conv-TasNet: surpassing ideal time-frequency magnitude masking for speech separation," IEEE TASLP, vol. 27, no. 8, pp. 1256-1266, 2019.

## Download dataset
Download LibriSpeech corpus with the script in asr task.
```bash
./download.sh <wham_download_dir> <librispeech_download_dir> <librimix_download_dir>
```

## Preprocess dataset
```bash
poetry run python preprocess.py --data_dir <librimix_download_dir> --out_dir data
```

## Train model
```bash
poetry run python -m konpeki.tasks.ss.train \
    --config-path=$(pwd) \
    --config-name="config" \
    dataset.train_json_path="data/train.json" \
    dataset.valid_json_path="data/valid.json" \
    train.out_dir="results" \
    train.checkpoint_path=
```

## Evaluate model
```bash
poetry run python -m konpeki.tasks.ss.evaluate \
    --config-path="$(pwd)" \
    --config-name="config" \
    dataset.test_json_path="data/test.json" \
    evaluate.out_dir="results" \
    evaluate.model_path="results/best_model.pt"
```
