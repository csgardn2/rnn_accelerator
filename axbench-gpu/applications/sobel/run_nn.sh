#!/bin/bash

# Regular Colors
black='\e[0;30m'        # Black
red='\e[0;31m'          # Red
green='\e[0;32m'        # Green
yellow='\e[0;33m'       # Yellow
blue='\e[0;34m'         # Blue
purple='\e[0;35m'       # Purple
cyan='\e[0;36m'         # Cyan
white='\e[0;37m'        # White

application=sobel

echo -e "${green} CUDA Sobel Edge-Detection Starting... ${white}"


for f in test.data/input/*.pgm
do
	filename=$(basename "$f")
	extension="${filename##*.}"
	filename="${filename%.*}"
	echo -e "-------------------------------------------------------"
	echo -e "${green} Input Image:  $f"
	echo -e "${green} output Image: ./train.data/output/${filename}_sobel.pgm ${white}"
	echo -e "-------------------------------------------------------"
	./bin/${application}_nn.out -file=$f -output=./test.data/output/${filename}_${application}_nn.pgm
	./bin/${application}.out    -file=$f -output=./test.data/output/${filename}_${application}.pgm
	compare -metric RMSE ./test.data/output/${filename}_${application}_nn.pgm ./test.data/output/${filename}_${application}.pgm null > tmp.log 2> tmp.err
	echo -ne "${red}$f\t"
	awk '{ printf("*** Error: %0.2f%\n",substr($2, 2, length($2) - 2) * 100) }' tmp.err
	echo -ne "${white}"
done
