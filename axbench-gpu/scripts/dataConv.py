#!/bin/usr/python

# Amir Yazdanbakhsh
# Jan. 7, 2015


#Format for reading binary file: (all sizes in bytes)
	#size of function name
	#name of the function
	#input or output (represented by 0 or 1)
	#type (given as integer, look at hadi's code for table)
	#value
	#range a
	#range b

import sys
import codecs
import struct
import os
import json
import re

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def printUsage():
	print "python dataConv.py <folder name>"
	exit(1)

def main():

	if(len(sys.argv) != 2):
		printUsage()

	folder_name = sys.argv[1]
	ioLen = 0
	n_inputs = 0
	n_outputs = 0

	output_data = folder_name + "/../fann.data/aggregated.fann"
	

	# first read the json file
	jsonFile=open("kernelNames.json").read()
	kernelNames = json.loads(jsonFile)

	for kernel in kernelNames:
		# find all data for each kernel
		totalData = []
		numInputs = 0
		numOutputs = 0
		totalDataCount = 0
		for file in os.listdir(folder_name):
			total_path = folder_name + "/" + file
			fileName, fileExtension = os.path.splitext(total_path)
			isTargetFile = re.match(r'kernel_' + kernel + '_.*\.data', file)
			if(isTargetFile):
				nInputs = int(kernelNames[kernel]['nInputs'])
				nOutputs = int(kernelNames[kernel]['nOutputs'])
				numInputs = nInputs
				numOutputs = nOutputs
				with open(total_path, 'r') as currFile:
					lines = currFile.readlines()
					lines.pop(0)
					for line in lines:
						line = line.rstrip()
						currDataSplitted = line.split(' ')
						currData = ''
						for i in range(nInputs):
							currData += currDataSplitted[i] + ' '
						pass
						currData += '\n'
						for j in range(nInputs, nInputs + nOutputs, 1):
							currData += currDataSplitted[j] + ' '
						pass
						currData += '\n'
						totalData.append(currData)
						totalDataCount += 1
					pass
				pass
		pass
		print "---------------------------------------------------------",
		print bcolors.UNDERLINE + "\n# Total number of training data (Kernel = %s): %d" % (kernel, len(totalData)) + bcolors.ENDC
		print "---------------------------------------------------------"
		with open(folder_name + "/../fann.data/" + kernel + "_aggregated.fann", "w") as outputFile:
			outputFile.write("%d %d %d\n" % (len(totalData), numInputs, numOutputs))
			for i in range(len(totalData)):
				outputFile.write(totalData[i])
			outputFile.close()
		pass

		activationConfig = folder_name + "/../../../fann.config/activationFunc_" + kernel + ".cfg"
		inputRangeA  = float(kernelNames[kernel]['InputRangeA'])
		inputRangeB  = float(kernelNames[kernel]['InputRangeB'])
		outputRangeA = float(kernelNames[kernel]['OutputRangeA'])
		outputRangeB = float(kernelNames[kernel]['OutputRangeB'])

		with open(activationConfig, 'w') as cfg:
			if(((outputRangeA >= 0.0) and (outputRangeA <= 1.0)) and ((outputRangeB >= 0.0) and (outputRangeB <= 1.0))):
				cfg.write("3") # sigmoid activation function
			elif(((outputRangeA > -1.0) and (outputRangeA < 1.0)) and ((outputRangeB > -1.0) and (outputRangeB < 1.0))):
				cfg.write("5") # sigmoid symmetric activation function
			else:
				cfg.write("0") # linear activation function
		pass
	pass
	


	# # first find all the files in the folder
	# input = []
	# output = []
	# dataSize = 0
	# for file in os.listdir(folder_name):
	# 	total_path = folder_name + "/" + file
	# 	fileName, fileExtension = os.path.splitext(total_path)
	# 	if(fileExtension != ".bin"):
	# 		continue

	# 	typeDict 	= {0:1, 1:1, 2:4, 3:4, 4:8}
	# 	name  		= total_path
	# 	fileSize    = os.path.getsize(name)

	# 	with open(name, "rb") as inf:
	# 		x = 0
	# 		while x < fileSize:
	# 			size = struct.unpack('i', inf.read(4))[0]
	# 			x += 4
	# 			#print "{} {}".format('Function Name Size:',size) #size of the function name
	# 			name = inf.read(size)
	# 			#print "{} {}".format('Function Name:', name) #name of the function
	# 			x += size
	# 			io = struct.unpack('b', inf.read(1))[0]
	# 			#print "{} {}".format('Input or Output:',io) #input or output
	# 			x += 1

	# 			# update the lenghth of inputs and outputs
	# 			ioLen = struct.unpack('i', inf.read(4))[0]
	# 			x += 4

	# 			type = struct.unpack('b', inf.read(1))[0]
	# 			#print "{} {}".format('Type:',type) #type (boolean, int, etc.) decide next read size based on that
	# 			x += 1

	# 			if(io == 0):
	# 				n_inputs = ioLen
	# 				for in_size in range(n_inputs):
	# 					if type == 0:
	# 						value = struct.unpack('?', inf.read(1))[0]
	# 						#print "{} {}".format('Value:',value)
	# 					elif type == 1:
	# 						value = struct.unpack('c', inf.read(1))[0]
	# 						#print "{} {}".format('Value:',value)
	# 					elif type == 2:
	# 						value = struct.unpack('i', inf.read(4))[0]
	# 						#print "{} {}".format('Value:',value)
	# 					elif type == 3:
	# 						value = struct.unpack('I', inf.read(4))[0]
	# 						#print "{} {}".format('Value:',value)
	# 					elif type == 4:
	# 						value = struct.unpack('d', inf.read(8))[0]
	# 						#print "{} {}".format('Value:',value) #value
	# 					x += typeDict.get(type)
	# 					input.append(value)
	# 				dataSize += 1
	# 			else:
	# 				n_outputs = ioLen
	# 				for in_size in range(n_outputs):
	# 					if type == 0:
	# 						value = struct.unpack('?', inf.read(1))[0]
	# 						#print "{} {}".format('Value:',value)
	# 					elif type == 1:
	# 						value = struct.unpack('c', inf.read(1))[0]
	# 						#print "{} {}".format('Value:',value)
	# 					elif type == 2:
	# 						value = struct.unpack('i', inf.read(4))[0]
	# 						#print "{} {}".format('Value:',value)
	# 					elif type == 3:
	# 						value = struct.unpack('I', inf.read(4))[0]
	# 						#print "{} {}".format('Value:',value)
	# 					elif type == 4:
	# 						value = struct.unpack('d', inf.read(8))[0]
	# 						#print "{} {}".format('Value:',value) #value
	# 					x += typeDict.get(type)
	# 					output.append(value)
	# 			rangeA = struct.unpack('d', inf.read(8))[0]
	# 			rangeB = struct.unpack('d', inf.read(8))[0]
	# 			x += 16

	# with open(activationConfig, 'w') as cfg:
	# 	if(((rangeA >= 0.0) and (rangeA < 1.0)) and ((rangeB >= 0.0) and (rangeB < 1.0))):
	# 		cfg.write("3") # sigmoid activation function
	# 	elif(((rangeA > -1.0) and (rangeA < 1.0)) and ((rangeB > -1.0) and (rangeB < 1.0))):
	# 		cfg.write("5") # sigmoid symmetric activation function
	# 	else:
	# 		cfg.write("0") # linear activation function
				
	# with open(output_data, 'w') as outf:
	# 	print "---------------------------------------------------------",
	# 	print bcolors.UNDERLINE + "\n# Total number of training data: %d" % (dataSize) + bcolors.ENDC
	# 	print "---------------------------------------------------------"
	# 	outf.write("{} {} {}".format(dataSize, n_inputs, n_outputs) + '\n')
	# 	for x in range(0, dataSize):
	# 		currInput = ""
	# 		for x in range(0,n_inputs):
	# 			currInput = currInput + " " + str(input.pop(0))
	# 		outf.write("{}\n".format(currInput))
	# 		currOutput = ""
	# 		for x in range(0,n_outputs):
	# 			currOutput = currOutput + " " + str(output.pop(0))
	# 		outf.write("{}\n".format(currOutput))
	# outf.close()
	# cfg.close()

if __name__ == "__main__":
    main()