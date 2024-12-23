#!/usr/bin/env bash

if [ $# -ne 1 ]; then
  echo "Usage: $0 <out_dir>"
  exit 1
fi

out_dir=$1
base_url="https://openslr.trmal.net/resources/11"
file="librispeech-lm-norm.txt.gz"

mkdir -p ${out_dir}

file_path=${out_dir}/${file}
if [ ! -e ${file_path} ]; then
  echo "Download ${file} into ${out_dir}."
  wget -O ${file_path} ${base_url}/${file}
  echo "Extract ${file} into ${out_dir}."
  gzip -d -k ${file_path}
fi

echo "Complete successfully."
