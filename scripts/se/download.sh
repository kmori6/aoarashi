#!/usr/bin/env bash

if [ $# -ne 1 ]; then
  echo "Usage: $0 <out_dir>"
  exit 1
fi

download_dir=$1
base_url="https://datashare.ed.ac.uk/bitstream/handle/10283/2791"
files="noisy_trainset_28spk_wav.zip clean_trainset_28spk_wav.zip noisy_testset_wav.zip clean_testset_wav.zip"

mkdir -p ${download_dir}

for file in ${files}; do
  file_path=${download_dir}/${file}
  if [ ! -e ${file_path} ]; then
    echo "Download ${file} into ${download_dir}."
    wget -O ${file_path} ${base_url}/${file}
  fi
  echo "Extract ${file} into ${download_dir}."
  unzip ${file_path} -d ${download_dir}
done

echo "Complete successfully."
