#!/bin/usr/python

# Amir Yazdanbakhsh
# Jan. 7, 2015

import os
import json
import sys
import subprocess

jsonFile=open("kernelNames.json").read()
kernelNames = json.loads(jsonFile)

with open("kernelNames.tmp" , "w") as outFile:
	for kernel in kernelNames:
		outFile.write(kernel)
		outFile.write("\n")
outFile.close() 