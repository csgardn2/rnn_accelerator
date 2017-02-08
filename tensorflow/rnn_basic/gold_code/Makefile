COMPILER=g++
EXENAME=training_data_generator
COMPILEFLAGS=-Wall -Wextra -ansi -Wfatal-errors -std=c++11
LINKFLAGS=-Wall -Wextra -ansi -Wfatal-errors
LIBS=
ARGS=65536 training_inputs.py

all: documentation $(EXENAME)

# add additional .o files on the line below (after main.o)
$(EXENAME): main.o
	$(COMPILER) $(LINKFLAGS) -o $(EXENAME) $^ $(LIBS)
	@echo "*** COMPILE_SUCCESSFUL ***"

#########################

# add more .cpp -> .o compile commands here

main.o: main.cpp
	$(COMPILER) $(COMPILEFLAGS) -c -o $@ $<

Doxyfile:
	@echo "Couldn't find Doxyfile, generating default"
	doxygen -g

#########################

clean:
	@rm -fv *.o $(EXENAME)

run: $(EXENAME)
	./$(EXENAME) $(ARGS)

valgrind: $(EXENAME)
	valgrind --leak-check=full ./$(EXENAME) $(ARGS)

documentation: Doxyfile $(ls *.cpp *.h)
	doxygen Doxyfile

