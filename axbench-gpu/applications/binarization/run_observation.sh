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


echo -e "${Green} CUDA Binarization Starting... ${White}"

if [ ! -d ./train.data/output/kernel.data ]; then
	mkdir ./train.data/output/kernel.data
fi

for f in train.data/input/*.bmp
do
	filename=$(basename "$f")
	extension="${filename##*.}"
	filename="${filename%.*}"
	echo -e "-------------------------------------------------------"
	echo -e "${Green} Input Image:  $f"
	echo -e "${Green} output Image: ./train.data/output/${filename}_bin.pgm ${White}"
	echo -e "-------------------------------------------------------"
	./bin/binarization.out $f ./train.data/output/${filename}_bin.pgm
	
	for kData in ./kernel_*.data
	do
		filename_kernel=$(basename "$kData")
		extension_kernel="${filename_kernel##*.}"
		filename_kernel="${filename_kernel%.*}"
		mv ${kData} ./train.data/output/kernel.data/${filename_kernel}_${filename}.data
	done
done
