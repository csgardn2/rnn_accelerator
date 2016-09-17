#!/bin/bash

# Regular Colors
Black='\e[0;30m'        # Black
Red='\e[0;31m'          # Red
Green='\e[0;32m'        # Green
Yellow='\e[0;33m'       # Yellow
Blue='\e[0;34m'         # Blue
Purple='\e[0;35m'       # Purple
Cyan='\e[0;36m'         # Cyan
White='\e[0;37m'        # White


echo -e "${Green} CUDA SRAD Starting... ${White}"

if [ ! -d ./train.data/output/kernel.data ]; then
	mkdir ./train.data/output/kernel.data
fi

./bin/srad.out 100 0.5 502 458 ./train.data/input/image.pgm ./train.data/output/image_out.pgm

for kData in ./kernel_*.data
do
	filename_kernel=$(basename "$kData")
	extension_kernel="${filename_kernel##*.}"
	filename_kernel="${filename_kernel%.*}"
	mv ${kData} ./train.data/output/kernel.data/${filename_kernel}_${filename}.data
done

