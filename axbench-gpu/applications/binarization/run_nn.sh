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


application=binarization

echo -e "${Green} CUDA Binarization Starting... ${White}"

for f in test.data/input/*.bmp
do
	filename=$(basename "$f")
	extension="${filename##*.}"
	filename="${filename%.*}"
	echo -e "-------------------------------------------------------"
	echo -e "${Green} Input Image:  $f"
	echo -e "${Green} output Image: ./train.data/output/${filename}_bin.pgm ${White}"
	echo -e "-------------------------------------------------------"
	./bin/${application}_nn.out 	$f ./test.data/output/${filename}_bin_nn.pgm
	./bin/${application}.out 		$f ./test.data/output/${filename}_bin.pgm
	compare -metric RMSE ./test.data/output/${filename}_bin_nn.pgm ./test.data/output/${filename}_bin.pgm null > tmp.log 2> tmp.err
	echo -ne "${Red}$f\t"
	awk '{ printf("*** Error: %0.2f%%\n",substr($2, 2, length($2) - 2) * 100) }' tmp.err
	echo -ne "${White}"
done
