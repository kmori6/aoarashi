# Text-to-Speech
Unofficial implementation based on D. Lim et al., "E2E-S2S-VC: end-to-end sequence-to-sequence voice conversion," in *Interspeech*, 2023, pp. 2043-2047.

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
poetry run python tokenizer.py --text_path "data/train.txt" --out_dir "results"
```

## Train model
```bash
poetry run python -m konpeki.tasks.vc.train \
    --config-path=$(pwd) \
    --config-name="config" \
    dataset.train_json_path="data/train.json" \
    dataset.valid_json_path="data/valid.json" \
    train.out_dir="results" \
    train.checkpoint_path=
```

## Evaluate model
```bash
poetry run python -m konpeki.tasks.vc.evaluate \
    --config-path="$(pwd)" \
    --config-name="config" \
    dataset.test_json_path="data/test.json" \
    evaluate.out_dir="results" \
    evaluate.model_path="<checkpoint_path>"
```

## Infer model
```bash
poetry run python -m konpeki.tasks.vc.infer \
    --config "config.yaml" \
    --out_dir "results" \
    --checkpoint_path <checkpoint_path> \
    --wav_path <wav_path>
```
