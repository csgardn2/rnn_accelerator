#!/bin/usr/python

# Amir Yazdanbakhsh
# Jan. 7, 2015

import os
import sys
import shutil
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
    print "python parrotConv.py <bench name>"
    exit(1)

def findFileName(filename):
    matchObj = re.match( r'()', filename)
pass;

def listCppFiles(dir):
    fileList = []
    extList  = []
    for f in os.listdir(dir):
        if(".c" in f):
            extLoc = f.rfind(".")
            fileList.append(f[:-(len(f)-extLoc)])
            extList.append(f[extLoc+1:])
    return (fileList, extList)
pass;

def parseParrotArgs(args):
        args = re.sub(',\s+', ',', args)
        args = re.sub('\s+$', '', args)
        args = args.split(',')

        parrotArgs = []
        for a in args:
            #print '-'*32
            #print a
            m = re.match('^\s*(\[(.+)\])?\s*(<(.+);(.+)>)?\s*(.+)$', a)

            features = [m.group(6), m.group(2), m.group(4), m.group(5)]
            #print features
            for i in range(len(features)):
                if ((features[i] == None) and (i != 1)):
                    features[i] = '0'
                pass
            
                if (features[i] == None):
                    continue
            
                features[i] = re.sub('^\s+', '', features[i])
                features[i] = re.sub('\s+$', '', features[i])
            pass

            parrotArgs.append((features[0], features[1], (features[2], features[3])))
            #print parrotArgs[-1]
        pass
    
        return parrotArgs
pass;

def parseParrotPragma(line, keyword):
    # process pragma
    m = re.match(
                    '#pragma\s+' +
                    'parrot' +
                    '\s*\(\s*' +
                    keyword +
                    '\s*,\s*' +
                    '"(.+)"' +
                    '\s*,\s*' +
                    '(.+)\)',
                    line
                )
                    
    parrotArgs = parseParrotArgs(m.group(2))
    parrotArgs.append(m.group(1))
    return parrotArgs
pass;

def listKernelNN(path):
    fileList = []
    extList  = []
    for f in os.listdir(path):
        if(".nn" in f):
            extLoc = f.rfind(".")
            fileList.append(f[:-(len(f)-extLoc)])
            extList.append(f[extLoc+1:])
    return (fileList, extList)
pass

def main():

    bench_name = sys.argv[1]

    # search src directory for all the c | cpp | cu files
    src_dir = os.getcwd() + "/src"
    obj_dir = os.getcwd() + "/obj"
    nn_dir  = os.getcwd() + "/src.nn"
    fann_dir = os.getcwd() + "/fann.config"

    (fileList, extList) = listCppFiles(src_dir)
    (kernelList, kernelExtList) = listKernelNN(fann_dir)


    for i in range(len(fileList)):
        fileStr = ""
        parrotoFile = fileList[i] + "." + extList[i]
        startPargma = False
        hasParrot = False

        currFile = open(src_dir + "/" + parrotoFile)
        lines = currFile.readlines()

        for line in lines:
            line = line.rstrip()

            # start of pragma
            if(("#pragma parrot" in line) and not startPargma and ("parrot.start" not in line) and ("parrot.end" not in line)):

                (varName, varLen, varRange) = parseParrotPragma(line, 'input')[0]
                hotFuncName =  parseParrotPragma(line, 'input')[1]

                # check if txt file for this hotfunc exists
                if(os.path.isfile(fann_dir + "/" + hotFuncName + "_cuda.txt")):
                    hotFuncFile = open(fann_dir + "/" + hotFuncName + "_cuda.txt")
                    lines = hotFuncFile.readlines()
                    for line in lines:
                        line = line.rstrip()
                        fileStr += line + "\n\n"
                    pass
                pass

                fileStr += "%s %s\n" % ("//", line)

                startPargma = True
                hasParrot = True
                continue
            if(("#pragma parrot" in line) and startPargma and ("parrot.start" not in line) and ("parrot.end" not in line)): # start of pragma
                startPargma = False
                fileStr += "%s %s\n" % ("//", line)
                (varName, varLen, varRange) = parseParrotPragma(line, 'output')[0]
                continue

            # check if we are between pragmas
            if(startPargma):
                fileStr += "%s %s\n" % ("//", line)
            else:
                fileStr += line + "\n"

        dstFilename   = fileList[i] + "_nn." + extList[i]
        dstFileHandle = open(nn_dir + "/" + dstFilename, "w")
        if(hasParrot):
            fileStr = "#include \"../../../headers/activationFunction.h\"\n\n" + fileStr
        dstFileHandle.write(fileStr) 

if __name__ == "__main__":
    main()
