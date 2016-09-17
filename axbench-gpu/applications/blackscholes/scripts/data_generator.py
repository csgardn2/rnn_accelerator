#!/usr/bin/python

# Designed by: Amir Yazdanbakhsh
# Date: March 26th - 2015
# Alternative Computing Technologies Lab.
# Georgia Institute of Technology

import sys
import random
import math

def Usage():
	print "Usage: python data_generator.py <size> <output file>"
	exit(1)

if(len(sys.argv) != 3):
	Usage()

data_size 		= sys.argv[1] 
output_file 	= open(sys.argv[2], 'w')

output_file.write(str(data_size) + "\n")

for i in range(int(data_size)):
	# Generate x and y target coordinates as random floats from -NUM_JOINTS to NUM_JOINTS
	stockPrice		= random.uniform(5.0,30.0)
	optionStrike 	= random.uniform(50.0,100.0);
	optionYear 		= random.randint(5,10)
	output_file.write("%f %f %d\n" % (stockPrice, optionStrike, optionYear))
pass;

print "Thank you..."