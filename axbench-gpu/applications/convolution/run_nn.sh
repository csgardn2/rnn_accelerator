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


echo -e "${Green} CUDA Convolution Separable Starting... ${White}"

for f in test.data/input/*.pgm
do
	filename=$(basename "$f")
	extension="${filename##*.}"
	filename="${filename%.*}"
	echo -e "-------------------------------------------------------"
	echo -e "${Green} Input Image:  $f${White}"
	echo -e "-------------------------------------------------------"
	./bin/convolutionSeparable_nn.out $f > error.tmp
	echo -ne "${Red}$f\t"
	awk '{ printf("*** Error: %0.2f%%\n",$1)}' error.tmp
	echo -ne "${White}"
done
