#!/usr/bin/env bash

if [ $# -ne 1 ]; then
  echo "Usage: $0 <out_dir>"
  exit 1
fi

out_dir=$1
base_url="https://openslr.trmal.net/resources/12"
files="train-clean-100.tar.gz train-clean-360.tar.gz train-other-500.tar.gz dev-clean.tar.gz dev-other.tar.gz test-clean.tar.gz test-other.tar.gz"

mkdir -p ${out_dir}

for file in ${files}; do
  file_path=${out_dir}/${file}
  if [ ! -e ${file_path} ]; then
    echo "Download ${file} into ${out_dir}."
    wget -O ${file_path} ${base_url}/${file}
  fi
  echo "Extract ${file} into ${out_dir}."
  tar -xzf ${file_path} -C ${out_dir}
done

echo "Complete successfully."
