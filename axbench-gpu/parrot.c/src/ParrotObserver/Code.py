'''
Created on Jul 22, 2012

@author: hadianeh
'''

import re
import os
import json
import os.path

class Code(object):
    PRAGMA_PARROT_KEYWORD = 'parrot'
    PRAGMA_PARROT_INPUT_KEYWORD = 'input'
    PRAGMA_PARROT_OUTPUT_KEYWORD = 'output'

    def __init__(self):
        self.src = ''
        self.regions = []
        self.kernelNames = {}
        self.parrotRegions = {}
        self.name = ''
        self.type = ''
        self.tempFiles = []
    pass

    # #pragma parrot.begin
    # #pragma parrot.end
    # These kewords go into the start and end of the C function that calls cuda kernel
    def parseParrotHostRegionsPragma(self, line, keyword):
        regularExpression = r'#pragma\s+parrot.' + keyword + '\s*\(\s*' + '(".+")\s*\)'
        parrotMatch = re.match(regularExpression, line)
        try:
            parrotArrays = parrotMatch.group(1)
            parrotList   = re.findall('"(.+?)",*' , parrotArrays)
            return [True, parrotList]
        except:
            return [False, None]
        pass
    pass


    # #pragma parrot(input/output, "ParrotName",
    # ([expression])?(<expression, expression>)? expression 
    # (, ([expression])?(<expression, expression>)? expression)* )
    def parseParrotPragma(self, line, keyword):
        parrotName = None
        parrotArgs = None
        
        m = re.match(
            '#pragma\s+' +
            self.PRAGMA_PARROT_KEYWORD +
            '\s*\(\s*' +
            keyword +
            '\s*,\s*' +
            '"(.+)"' +
            '\s*,\s*' +
            '(.+)\)',
            
            line
        )
        try:
            parrotName = m.group(1)
            parrotArgs = self.parseParrotArgs(m.group(2))
            return [True, parrotName, parrotArgs]
        except:
            return [False, parrotName, parrotArgs]
        pass
    pass

    def parseParrotArgs(self, args):
        args = re.sub(',\s+', ',', args)
        args = re.sub('\s+$', '', args)
        args = args.split(',')

        parrotArgs = []
        for a in args:
            print '-'*32
            print a
            m = re.match('^\s*(\[(.+)\])?\s*(<(.+);(.+)>)?\s*(.+)$', a)

            features = [m.group(6), m.group(2), m.group(4), m.group(5)]
            print features
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
            print parrotArgs[-1]
        pass
    
        return parrotArgs
    pass

    def buildLine(self, src, i):
        line = src[i]
        m = re.match(
            '#pragma\s+' +
            self.PRAGMA_PARROT_KEYWORD +
            '.+(\\\\)\s*$',
            line
        )

        j = i + 1
        while(m != None):
            if (j < len(src)):
                line = line[:m.start(1)] + src[j] + line[m.end(1):]
            pass
        
            m = re.match(
                '.*(\\\\)\s*$',
                line
            )
            
            j += 1
        pass
    
        return line, j - 1
    pass

    def cppParser(self, srcFileName, extCmd, outFileName):

        print "cppParser"
        
        srcFile = open(self.tempFiles[-1])
        
        src = srcFile.readlines()
        self.src = src
        


        # find the host parrot region
        foundParrotBegin = False
        self.parrotHost = []
        inputLoc = 0
        inputParrotInfo = ()
        for i in range(len(src)):
            line, j = self.buildLine(src, i)
            
            if (not foundParrotBegin):
                inputParrotInfo = self.parseParrotHostRegionsPragma(line, 'start')
                if (inputParrotInfo[0]):
                    foundParrotBegin = True
                    inputLoc = (i, j) 
                pass
            else:
                outputParrotInfo = self.parseParrotHostRegionsPragma(line, 'end')
                if (outputParrotInfo[0]):
                    inputParrotInfo[0] = inputLoc
                    outputParrotInfo[0] = (i, j)
                    foundParrotBegin = False
                    self.parrotHost.append((inputParrotInfo, outputParrotInfo))
                pass
            pass
        pass

        print self.parrotHost


        # find the device parrot region
        foundParrotInput = False
        self.regions = []
        inputLoc = 0
        inputParrotInfo = ()
        
        for i in range(len(src)):
            line, j = self.buildLine(src, i)
            
            if (not foundParrotInput):
                inputParrotInfo = self.parseParrotPragma(line, self.PRAGMA_PARROT_INPUT_KEYWORD)
                if (inputParrotInfo[0]):
                    foundParrotInput = True
                    inputLoc = (i, j) 
                pass
            else:
                outputParrotInfo = self.parseParrotPragma(line, self.PRAGMA_PARROT_OUTPUT_KEYWORD)
                if (outputParrotInfo[0]):
                    if (inputParrotInfo[1] != outputParrotInfo[1]):
                        errMsg = 'Error: Oops! The Parrot names do not match on line '
                        errMsg += str(inputLoc + 1) + ' "' + inputParrotInfo[1] + '"' 
                        errMsg += ' and line ' + str(i + 1) + ' "' + outputParrotInfo[1] + '"!' 
                        
                        print errMsg    
                        return False
                    pass
                    inputParrotInfo[0] = inputLoc
                    outputParrotInfo[0] = (i, j)
                    foundParrotInput = False
                    self.parrotRegions[inputParrotInfo[1]] = {}
                    (inputRangeA,  inputRangeB)  = inputParrotInfo[2][0][2]
                    (outputRangeA, outputRangeB) = outputParrotInfo[2][0][2]
                    self.parrotRegions[inputParrotInfo[1]]['nInputs'] = inputParrotInfo[2][0][1]
                    self.parrotRegions[inputParrotInfo[1]]['nOutputs'] = outputParrotInfo[2][0][1]
                    self.parrotRegions[inputParrotInfo[1]]['InputRangeA']  = float(inputRangeA)
                    self.parrotRegions[inputParrotInfo[1]]['InputRangeB']  = float(inputRangeB)
                    self.parrotRegions[inputParrotInfo[1]]['OutputRangeA'] = float(outputRangeA)
                    self.parrotRegions[inputParrotInfo[1]]['OutputRangeB'] = float(outputRangeB)
                    print json.dumps(self.parrotRegions, sort_keys=False, indent=4, separators=(',', ': '))
                    self.regions.append((inputParrotInfo, outputParrotInfo))
                pass
            pass
        pass
        return True
    pass

    def cppProbes(self, cfg):

        src = self.src
        
        # Amir
        isParrot = False
        isDevice = False
        parrotVarList = []

        currentKernelList = {}
        if(os.path.isfile('kernelNames.json')):
            kernelNames = open('kernelNames.json', 'r')
            currentKernelList = json.load(kernelNames)

        #print json.dumps(currentKernelList, sort_keys=False, indent=4, separators=(',', ': '))
        self.parrotRegions = {key: value for (key, value) in (self.parrotRegions.items() + currentKernelList.items())}

        #self.parrotRegions = dict(self.parrotRegions.items() + currentKernelList)

        with open('kernelNames.json', 'w') as kFile:
           json.dump(self.parrotRegions, kFile, sort_keys = True, indent = 4)
        pass
           
        # Rima

        # host part
        for j, region in enumerate(self.parrotHost):
            isParrotStart = True
            for i in range(2):
                loc = region[i][0]
                varList = region[i][1]
                
                probeStr = ''
                for var in varList:
                    nInputs = self.parrotRegions[var]["nInputs"]
                    nOutputs = self.parrotRegions[var]["nOutputs"]
                    if(isParrotStart):                        
                        probeStr += '\tfloat* hArray_' + var + ' = new float[PARROT_SIZE];\n'
                        probeStr += '\tint hIndex_' + var + ' = 0;\n'
                        probeStr += '\tcudaMemcpyToSymbol(dIndex_' + var + ', &hIndex_' + var + ', sizeof(int));'
                        probeStr += '\tFILE *of_' + var + ";\n"
                        probeStr += '\tof_' + var + '  = fopen("kernel_' + var + '.data", "w");\n\n'
                        parrotVarList.append(var)
                    else:
                        probeStr += '\tcudaMemcpyFromSymbol(hArray_' + var + ', dArray_' + var + ', PARROT_SIZE * sizeof(float));\n'
                        probeStr += '\tcudaMemcpyFromSymbol(&hIndex_' + var + ', dIndex_' + var + ', sizeof(int));\n'
                        probeStr += '\tfprintf(of_' + var + ', "' + nInputs + ' ' + nOutputs + '\\n");\n '
                        probeStr += '\tfor(int i_' + var + ' = 0; ' + 'i_' + var + ' < hIndex_' + var + '; i_' + var + '++) {\n'
                        probeStr += '\t\tfor(int j_' + var + ' = 0; ' + 'j_' + var + ' < ' + str(int(nInputs) + int(nOutputs)) + '; j_' + var + '++) {\n'
                        probeStr += '\t\t\tfprintf(of_' + var + ', "%f ", ' + 'hArray_' + var + '[i_' + var + ' * ' + str(int(nInputs) + int(nOutputs)) + ' + j_' + var + ']);\n'
                        probeStr += '\t\t}\n'
                        probeStr += '\t\tfprintf(of_' + var + ', "\\n");\n'
                        probeStr += '\t}\n'
                        probeStr += '\tfclose(' + 'of_' + var + ');\n\n'
                pass
                isParrotStart = False
                src.insert(loc[i ^ 1] + 1 + j * 2, probeStr)
            pass
        pass

        # device part
        for j, region in enumerate(self.regions):
            #print region
            isLoop = True
            nInputs = 1
            nOutputs = 1
            varList1 = region[0][2]
            varList2 = region[1][2]

            nInputs  = varList1[0][1]
            nOutputs = varList2[0][1] 

            for i in range(2):
                loc = region[i][0]
                tag = region[i][1]
                varList = region[i][2]

                # if(i == 0):
                #     nInputs = varList[0][1]
                # else:
                #     nOutputs = varList[0][1]

                probeStr = ''
                for var in varList:
                    if(isLoop):
                        isLoop = False
                        isDevice = True
                        probeStr += '\tint currIndex_' + tag + '= atomicAdd(&(dIndex_' + tag + '), (int)1);\n\n'
                        #probeStr += '\tif(( currIndex_' + tag + ' * ' + str(int(nInputs) + int(nOutputs)) + ' + ' + str(int(nInputs) + int(nOutputs)) + '- 1) < PARROT_SIZE){\n'
                        #probeStr += '\tif(currIndex_' + tag + " < int(PARROT_SIZE / " + str(int(nInputs) + int(nOutputs)) + ")){\n"
                        probeStr += '\tif(currIndex_' + tag + ' < 10000){\n'
                        # First loop over the inputs
                        probeStr += '\t\tfor (int pIndex = 0; pIndex < '
                        probeStr += var[1]
                        probeStr += '; pIndex++){\n'
                        probeStr += '\t\t\tdArray_' + tag + '[currIndex_' + tag + ' * '
                        probeStr += str(int(nInputs) + int(nOutputs))
                        probeStr += ' + pIndex] = parrotInput[pIndex];\n'
                        probeStr += '\t\t}\n'
                        probeStr += '}\n'
                        probeStr += '\telse{\n'
                        probeStr += '\t\tatomicSub(&(dIndex_' + tag + '), (int)1);\n'
                        probeStr += '}\n'
                    else:
                        # First loop over the inputs
                        #probeStr += '\tif(( currIndex_' + tag + ' * ' + str(int(nInputs) + int(nOutputs)) + ' + ' + str(int(nInputs) + int(nOutputs)) + '- 1) < PARROT_SIZE){\n'
                        #probeStr += '\tif(currIndex_' + tag + " < int(PARROT_SIZE / " + str(int(nInputs) + int(nOutputs)) + ")){\n"
                        probeStr += '\tif(currIndex_' + tag + ' < 10000){\n'
                        probeStr += '\t\tfor (int pIndex = 0; pIndex < '
                        probeStr += var[1]
                        probeStr += '; pIndex++){\n'
                        probeStr += '\t\t\tdArray_' + tag + '[currIndex_' + tag + ' * '
                        probeStr += str(int(nInputs) + int(nOutputs))
                        probeStr += ' + '
                        probeStr += str(int(nInputs))
                        probeStr += ' + pIndex] = parrotOutput[pIndex];\n'
                        probeStr += '\t\t}\n'
                        probeStr += '\t}\n'
                        # probeStr += '\telse{\n'
                        # probeStr += '\t\tatomicSub(&(dIndex_' + tag + '), (int)1);\n'
                        # probeStr += '\t}\n'

                    isParrot = True
                pass
            
                src.insert(loc[i ^ 1] + 1 + j * 2, probeStr)
            pass
        pass
    
        probeStr = ''
        if(isDevice):
            probeStr += '#ifndef PARROT_SIZE\n'
            probeStr += '\t#define PARROT_SIZE 10000000\n'
            probeStr += '#endif\n\n'
            for var in parrotVarList:
                probeStr += '__device__ float dArray_' + var + '[PARROT_SIZE];\n'
                probeStr += '__device__ int dIndex_' + var + ';\n'
            pass
        else:
            probeStr += '#ifndef PARROT_SIZE\n'
            probeStr += '\t#define PARROT_SIZE 10000000\n'
            probeStr += '#endif\n\n'
            for var in parrotVarList:
                probeStr += 'extern __device__ float dArray_' + var + '[PARROT_SIZE];\n'
                probeStr += 'extern __device__ int dIndex_' + var + ';\n'
            pass
        src.insert(0, probeStr)
        return src
    pass
                    
    def cppCompiler(self, extCmd, outFileName):
        cmd = extCmd + ' ' + self.tempFiles[-1] + ' -o ' + outFileName
        print cmd
        os.system(cmd)
    pass 

    parsers = {
        'c':   cppParser,
        'cpp': cppParser,
        'C':   cppParser,
        'cu':  cppParser,
        'cuh': cppParser,
        'CU':  cppParser
    }
    
    probes = {
        'c':   cppProbes,
        'cpp': cppProbes,
        'C':   cppProbes,
        'cu':  cppProbes,
        'cuh': cppProbes,
        'CU':  cppProbes
    }

    compilers = {
        'c':   cppCompiler,
        'cpp': cppCompiler,
        'C':   cppCompiler,
        'cu':  cppCompiler,
        'cuh': cppCompiler,
        'CU':  cppCompiler

    }
    
    def parse(self, srcFileName, extCmd, outFileName):
        self.__init__()
        
        m = re.match('(.+)\.(.+)$', srcFileName)
        
        self.name = m.group(1)
        self.type = m.group(2)
        
        self.tempFiles.append(srcFileName)
        
        self.parsers[self.type](self, srcFileName, extCmd, outFileName)
    pass

    def insertProbes(self, cfg):
        return self.probes[self.type](self, cfg)
    pass

    def compile(self, extCmd, outFileName):
        return self.compilers[self.type](self, extCmd, outFileName)
    pass 

pass

if __name__ == '__main__':
    os.remove('kernelNames.json')
    codeRegions = Code()
    codeRegions.find('kooft.hot.cpp')
    
    exit(0)
pass
