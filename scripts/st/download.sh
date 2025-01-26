#!/usr/bin/env bash

if [ $# -ne 1 ]; then
  echo "Usage: $0 <out_dir>"
  exit 1
fi

out_dir=$1
file_url="https://dl.fbaipublicfiles.com/covost/covost_v2.en_ja.tsv.tar.gz"

mkdir -p ${out_dir}

file=$(echo ${file_url} | rev | cut -d '/' -f 1 | rev)
file_path=${out_dir}/${file}
if [ ! -e ${file_path} ]; then
  echo "Download ${file} into ${out_dir}."
  wget -O ${file_path} ${file_url}
fi
echo "Extract ${file} into ${out_dir}."
tar -xzf ${file_path} -C ${out_dir}

echo "Complete successfully."
