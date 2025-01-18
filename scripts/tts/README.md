# Text-to-Speech
Unofficial implementation based on D. Lim et al., "JETS: jointly training FastSpeech2 and HiFi-GAN for end to end text to speech," in *Interspeech*, 2022, pp. 21-25.

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
poetry run python ../../src/aoarashi/tasks/tts/train.py \
    --config-path=$(pwd) \
    --config-name="config" \
    dataset.train_json_path="data/train.json" \
    dataset.valid_json_path="data/valid.json" \
    tokenizer.model_path="results/token.txt" \
    train.out_dir="results" \
    train.checkpoint_path=
```

## Evaluate model
```bash
poetry run python ../../src/aoarashi/tasks/tts/evaluate.py \
    --config-path="$(pwd)" \
    --config-name="config" \
    dataset.test_json_path="data/test.json" \
    tokenizer.model_path="results/token.txt" \
    evaluate.out_dir="results" \
    evaluate.model_path="<checkpoint_path>"
```

## Infer model
```bash
poetry run python ../../src/aoarashi/tasks/tts/infer.py \
    --config "config.yaml" \
    --out_dir "results" \
    --checkpoint_path <checkpoint_path> \
    --token_path "results/token.txt" \
    --input_text <text>
```
