#!/usr/bin/env bash

if [ $# -ne 1 ]; then
  echo "Usage: $0 <out_dir>"
  exit 1
fi

out_dir=$1
base_url=http://festvox.org/cmu_arctic/cmu_arctic/packed
spks="bdl slt clb rms"

mkdir -p ${out_dir}

for spk in ${spks}; do
  file=cmu_us_${spk}_arctic-0.95-release.tar.bz2
  file_path=${out_dir}/${file}
  if [ ! -e ${file_path} ]; then
    echo "Download ${file} into ${out_dir}."
    wget -O ${file_path} ${base_url}/${file}
  fi
  echo "Extract ${file} into ${out_dir}."
  tar -xjf ${file_path} -C ${out_dir}
done

echo "Complete successfully."
