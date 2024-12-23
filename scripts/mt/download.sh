#!/usr/bin/env bash

if [ $# -ne 1 ]; then
  echo "Usage: $0 <out_dir>"
  exit 1
fi

out_dir=$1
file_urls="https://www.statmt.org/wmt13/training-parallel-europarl-v7.tgz \
  https://www.statmt.org/wmt13/training-parallel-commoncrawl.tgz \
  https://www.statmt.org/wmt14/training-parallel-nc-v9.tgz \
  https://www.statmt.org/wmt14/dev.tgz \
  https://www.statmt.org/wmt14/test-full.tgz"

mkdir -p ${out_dir}

for file_url in ${file_urls}; do
  file=$(echo ${file_url} | rev | cut -d '/' -f 1 | rev)
  file_path=${out_dir}/${file}
  if [ ! -e ${file_path} ]; then
    echo "Download ${file} into ${out_dir}."
    wget -O ${file_path} ${file_url}
  fi
  if [ ${file} == "training-parallel-commoncrawl.tgz" ]; then
    _out_dir=${out_dir}/training
  else
    _out_dir=${out_dir}
  fi
  echo "Extract ${file} into ${_out_dir}."
  tar -xzf ${file_path} -C ${_out_dir}
done

# test preprocessing
test_dir=${out_dir}/test-full
for file in $(ls ${test_dir}); do
  file_name=$(echo ${file} | rev | cut -d '.' -f 2- | rev)
  grep '<seg id="[0-9]*">' ${test_dir}/${file} | \
    sed 's/<\/seg>//g' | sed 's/<seg id="[0-9]*">//g' > ${test_dir}/${file_name}
done

echo "Complete successfully."
