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

./bin/srad.out    100 0.5 502 458 ./test.data/input/image.pgm ./test.data/output/image_out.pgm
./bin/srad_nn.out 100 0.5 502 458 ./test.data/input/image.pgm ./test.data/output/image_out_nn.pgm
compare -metric RMSE ./test.data/output/image_out.pgm ./test.data/output/image_out_nn.pgm null > tmp.log 2> tmp.err
echo -ne "${Red}"
awk '{ printf("*** Error: %0.2f%\n",substr($2, 2, length($2) - 2) * 100) }' tmp.err
echo -ne "${White}"

