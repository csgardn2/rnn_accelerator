#!/bin/usr/python

# Amir Yazdanbakhsh
# Jan. 7, 2015

# convert the fann output to a text file which can be used inside the CUDA kernel

import os
import os.path
import sys
import shutil
import re
import json

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
    print bcolors.FAIL + "python fann2kernel.py <input file>" + bcolors.ENDC
    exit(1)
pass

def extractLayerConfig(line):
	layerConfig = re.findall("([0-9]+)", line)
	if(len(layerConfig) != 0):
		return layerConfig
pass

def extractNeuronsConfig(line, layerConfig, nnJSON):

	neuronConfigs = re.findall("([0-9]+, [0-9]+, [-|+]*[0-9]*\.[0-9]*[e|E]*[+|-]*[0-9]*)", line)

	# input layer starts with zero
	neuronIndex = 0
	for neuronConfig in neuronConfigs:
		neuronName = "n_" + str(neuronIndex)
		neuronIndex += 1
		currentConfig = re.match("([0-9]+), ([0-9]+), ([-|+]*[0-9]*\.[0-9]*[e|E]*[+|-]*[0-9]*)", neuronConfig)
		nnJSON["neurons"][neuronName] = {}
		nnJSON["neurons"][neuronName]["num_inputs"] = currentConfig.group(1)
		nnJSON["neurons"][neuronName]["activation_function"] = currentConfig.group(2)
		nnJSON["neurons"][neuronName]["activation_steepness"] = currentConfig.group(3)
	return nnJSON
pass

def extractWeightsConfig(line, nnJSON):

	weightConfigs = re.findall("([0-9]+, [-|+]*[0-9]*\.[0-9]*[e|E]*[+|-]*[0-9]*)", line)
	neuronLastWeightIndex = {}

	for weightConfig in weightConfigs:
		currentConfig = re.match("([0-9]+), ([-|+]*[0-9]*\.[0-9]*[e|E]*[+|-]*[0-9]*)", weightConfig)
		neuronName    = "n_" + str(currentConfig.group(1))
		try:
			neuronLastWeightIndex[neuronName] += 1
		except:
			neuronLastWeightIndex[neuronName] = 0
		nnJSON["neurons"][neuronName]["w_" + str(neuronLastWeightIndex[neuronName])] = currentConfig.group(2)
	return nnJSON
pass


# activation function library
activationFunc = {
	'0': 'linear',
	'3': 'sigmoid',
	'5': 'symmetricSigmoid'
}



def main():
	if(len(sys.argv) != 2):
		printUsage()

	# extract the neural network configuration
	nnLayerConfig = []
	nnJSON = {}

	fannFile = open(sys.argv[1])
	baseName = os.path.splitext(sys.argv[1])[0]
	fannLines = fannFile.readlines()


	isLayerDone = False
	isNeuronDone = False
	isLayerAdded = False

	for line in fannLines:
		line = line.rstrip()
		
		# match with the layer config
		matchLayer  = re.match("layer_sizes=.*", line)
		if(matchLayer):
			nnLayerConfig = extractLayerConfig(line)
			isLayerDone = True
		pass

		if(not isLayerDone):
			continue

		# add layers to JSON
		if(not isLayerAdded):
			hIndex = 0
			nnJSON["layers"] = {}
			nnJSON["neurons"] = {}
			for i in range(len(nnLayerConfig)):
				if(i == 0):
					nnJSON["layers"]["input"] = nnLayerConfig[0]
				elif(i == len(nnLayerConfig) - 1):
					nnJSON["layers"]["output"] = nnLayerConfig[len(nnLayerConfig) - 1]
				else:
					nnJSON["layers"]["hidden_" + str(hIndex)] = nnLayerConfig[i]
					hIndex += 1
			isLayerAdded = True
		pass

		# match the neuron config
		matchNeuron = re.match("neurons\s*\(num_inputs, activation_function, activation_steepness\).*", line)
		if(matchNeuron):
			nnJSON = extractNeuronsConfig(line, nnLayerConfig, nnJSON)
			isNeuronDone = True
		pass

		# print json.dumps(nnJSON, sort_keys=True, indent=4, separators=(',', ': '))
		if(not isNeuronDone):
			continue

		# match the connection config
		matchConnection = re.match("connections\s*\(connected_to_neuron, weight\).*", line)
		if(matchConnection):
			nnJSON = extractWeightsConfig(line, nnJSON)
		pass
	pass

	# dump JSON of the NN
	with open(baseName + ".json", 'w') as outfile:
		json.dump(nnJSON, outfile, sort_keys=True, indent=4)


	kernelFile = open(baseName + "_cuda.txt", 'w')

	# first input layer
	inputLayerNeurons = int(nnLayerConfig[0])
	firstHiddenLayer  = nnLayerConfig[1]
	numberOfWeights   = int(firstHiddenLayer) - 1
	for i in range(numberOfWeights):
		currStr = "float layer_1_" + str(i) + " = "
		for j in range(inputLayerNeurons - 1): # bias is the last neuron
			currStr += "parrotInput[%d] * %f + " % (j, float(nnJSON["neurons"]["n_" + str(j)]["w_" + str(i)]))
		currStr += "1.0f * %f;\n" % (float(nnJSON["neurons"]["n_" + str(inputLayerNeurons-1)]["w_" + str(i)]))
		kernelFile.write(currStr)

	baseNeuronIndex = int(inputLayerNeurons)
	for i in range(2, len(nnLayerConfig)): # layer 2, 3, ...
		for j in range(int(nnLayerConfig[i]) - 1): # number of neurons in the current layer without bias
			currStr = "float layer_" + str(i) + "_" + str(j) + " = "
			for k in range(int(nnLayerConfig[i-1]) - 1): # extract the weights from all of the previous nodes
				currStr += "%s(layer_%d_%d, %f) * %f + " % (activationFunc[nnJSON["neurons"]["n_" + str(baseNeuronIndex + k)]["activation_function"]],  i-1, k, float(nnJSON["neurons"]["n_" + str(baseNeuronIndex + k)]["activation_steepness"]), float(nnJSON["neurons"]["n_" + str(baseNeuronIndex + k)]["w_" + str(j)]))
			pass
			currStr += "1.0f * %f;\n" % (float(nnJSON["neurons"]["n_" + str(baseNeuronIndex + int(nnLayerConfig[i-1]) - 1)]["w_" + str(j)]))
			kernelFile.write(currStr)
			currStr = "layer_%d_%d = %s(layer_%d_%d, %s);\n" % (i, j, activationFunc[nnJSON["neurons"]["n_" + str(baseNeuronIndex + int(nnLayerConfig[i-1]) + j)]["activation_function"]], i, j, float(nnJSON["neurons"]["n_" + str(baseNeuronIndex + int(nnLayerConfig[i-1]) + j)]["activation_steepness"]))
			kernelFile.write(currStr)
		baseNeuronIndex += int(nnLayerConfig[i])
		pass
	pass
	# copy output to parrot out
	outputNeurons = nnLayerConfig[len(nnLayerConfig)-1]
	for i in range(int(outputNeurons) - 1):
		currStr = "parrotOutput[%d] = layer_%d_%d;\n" % (i, len(nnLayerConfig) - 1, i)
		kernelFile.write(currStr)
	pass

	kernelFile.close()

pass



if __name__ == "__main__":
	main()