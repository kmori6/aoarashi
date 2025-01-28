#!/usr/bin/env bash

if [ $# -ne 3 ]; then
  echo "Usage: $0 <wham_download_dir> <librispeech_download_dir> <librimix_download_dir>"
  exit 1
fi

wham_download_dir=$1
librispeech_download_dir=$2
librimix_download_dir=$3

# experiment settings
num_speakers=2
sample_rate=8k
mode=min

# WHAM! download
mkdir -p ${wham_download_dir}
wham_file_url="https://my-bucket-a8b4b49c25c811ee9a7e8bba05fa24c7.s3.amazonaws.com/wham_noise.zip"
wham_file=$(echo ${wham_file_url} | rev | cut -d '/' -f 1 | rev)
wham_file_path=${wham_download_dir}/${wham_file}
if [ ! -e ${wham_file_path} ]; then
  echo "Download ${wham_file} into ${wham_download_dir}."
  wget -O ${wham_file_path} ${wham_file_url}
fi
echo "Extract ${file} into ${wham_download_dir}."
unzip ${wham_file_path} -d ${wham_download_dir}

# LibriMix download
mkdir -p ${librimix_download_dir}
git clone https://github.com/JorisCos/LibriMix ${librimix_download_dir}/LibriMix
# execute https://github.com/JorisCos/LibriMix/blob/master/generate_librimix.sh manually for minimum sample generation
poetry run python ${librimix_download_dir}/LibriMix/scripts/augment_train_noise.py --wham_dir ${wham_download_dir}/wham_noise
poetry run python ${librimix_download_dir}/LibriMix/scripts/create_librimix_from_metadata.py \
  --librispeech_dir ${librispeech_download_dir} \
  --wham_dir ${wham_download_dir}/wham_noise \
  --metadata_dir ${librimix_download_dir}/LibriMix/metadata/Libri${num_speakers}Mix \
  --librimix_outdir ${librimix_download_dir} \
  --n_src ${num_speakers} \
  --freqs ${sample_rate} \
  --modes ${mode} \
  --types mix_clean mix_both mix_single

echo "Complete successfully."
