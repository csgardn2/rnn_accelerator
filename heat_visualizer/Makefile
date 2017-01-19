COMPILER=g++
EXENAME=heat_visualizer
COMPILEFLAGS=-Wall -Wextra -ansi -Wfatal-errors -std=c++11
LINKFLAGS=-Wall -Wextra -ansi -Wfatal-errors
LIBS=-lpng
ARGS=temp_512 output.png

all: documentation $(EXENAME)

# add additional .o files on the line below (after main.o)
$(EXENAME): main.o png.o utility.o
	$(COMPILER) $(LINKFLAGS) -o $(EXENAME) $^ $(LIBS)
	@echo "*** COMPILE_SUCCESSFUL ***"

#########################

# add more .cpp -> .o compile commands here

main.o: main.cpp
	$(COMPILER) $(COMPILEFLAGS) -c -o $@ $<

png.o: png.cpp png.h
	$(COMPILER) $(COMPILEFLAGS) -c -o $@ $<

utility.o: utility.cpp utility.h
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

