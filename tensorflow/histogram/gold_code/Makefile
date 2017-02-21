COMPILER=g++
EXENAME=training_data_generator
COMPILEFLAGS=-g -Wall -Wextra -ansi -Wfatal-errors -std=c++11
LINKFLAGS=-g -Wall -Wextra -ansi -Wfatal-errors
LIBS=
ARGS=16384 1024 64 input_samples.py output_histograms.py
# Usage:
#     ./generate_training_data
#         [number of training samplea]
#         [number of input elements per sample]
#         [number of histogram output bins per sample]
#         [filename to write input samples to]
#         [filename to write output histograms to]

all: documentation $(EXENAME)

# add additional .o files on the line below (after main.o)
$(EXENAME): generate_training_data.o
	$(COMPILER) $(LINKFLAGS) -o $(EXENAME) $^ $(LIBS)
	@echo "*** COMPILE_SUCCESSFUL ***"

#########################

# add more .cpp -> .o compile commands here

generate_training_data.o: generate_training_data.cpp
	$(COMPILER) $(COMPILEFLAGS) -c -o $@ $<

Doxyfile:
	@echo "Couldn't find Doxyfile, generating default"
	doxygen -g

#########################

clean:
	@rm -fvr *.o $(EXENAME)

run: $(EXENAME)
	./$(EXENAME) $(ARGS)

# Suppressions seem to be assiciated with the standard library.
# Proof: The following empty C++ main function produces memory leaks on my
# machine
#
# include <iostream>
# int main(int argc, char** argv)
# {
#     return 0;
# }
#
valgrind: $(EXENAME)
	valgrind --suppressions=valgrind_suppressions.supp --leak-check=full ./$(EXENAME) $(ARGS)

documentation: Doxyfile $(ls *.cpp *.h)
	doxygen Doxyfile

