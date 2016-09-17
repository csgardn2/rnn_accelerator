#!/bin/bash

# Created by Amir Yazdanbakhsh on 1/14/16
# a.yazdanbakhsh@gatech.edu

# include files
. ../include/bash_color.h
. ../include/gpgpu_sim.mk

# Global Configurations
BENCHMARK=laplacian
SIMULATION_TYPE=orig_code
BIN=${BENCHMARK}_${SIMULATION_TYPE}.out

Usage()
{
	echo -e "${Red}./run_baseline.sh${White}"
}

if [ "$#" -ge 1 ]; then
	if [ "$1" == "--help" ]; then
		Usage
	else
		echo -e "${Red}Use --help to learn how to use this bash script.${White}"
	fi
else
	echo -e "${Blue}- Create the log directory...${White}"
		if [ ! -d log ]; then
			mkdir log
		fi

	echo -e "${Blue}- Make the source file...${White}"
		make clean > /dev/null
		make SIM_TYPE=${SIMULATION_TYPE} > ./log/make.log 2>&1
		if [ "$?" -ne 0 ]; then
			echo -e"${Red} Build failed...${White}"
			cat ./log/make.log
			exit
		fi
	
	echo -e "${Blue}- Run the code...${White}"
		for f in ./data/input/*.pgm
		do
			filename=$(basename "$f")
			extension="${filename##*.}"
			filename="${filename%.*}"
			
			./bin/${BIN} -file=$f -output=./data/output/${filename}_output.pgm > ./log/${filename}.log
		done
	echo -e "${Green} Thanks for using AxBench-GPU...${White}"
fi