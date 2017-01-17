COMPILER=g++
EXENAME=gradient
COMPILEFLAGS=-Wall -Wextra -ansi -Wfatal-errors -std=c++11
LINKFLAGS=-Wall -Wextra -ansi -Wfatal-errors
LIBS=-lpng
ARGS=citadel_input.png output.png

all: $(EXENAME)

# add additional .o files on the line below (after main.o)
$(EXENAME): gradient.o main.o png.o tile_linear_gold.o tile_linear_neural_01_sucky.o tile_linear_neural_02_basic.o tile_nonlinear_gold.o tile_nonlinear_neural_01_sucky.o
	$(COMPILER) $(LINKFLAGS) -o $(EXENAME) $^ $(LIBS)
	@echo "*** COMPILE_SUCCESSFUL ***"

#########################

# add more .cpp -> .o compile commands here

gradient.o: gradient.cpp
	$(COMPILER) $(COMPILEFLAGS) -c -o $@ $<

main.o: main.cpp
	$(COMPILER) $(COMPILEFLAGS) -c -o $@ $<

png.o: png.cpp png.h
	$(COMPILER) $(COMPILEFLAGS) -c -o $@ $<

tile_linear_gold.o: tile_linear_gold.cpp
	$(COMPILER) $(COMPILEFLAGS) -c -o $@ $<

tile_linear_neural_01_sucky.o: tile_linear_neural_01_sucky.cpp
	$(COMPILER) $(COMPILEFLAGS) -c -o $@ $<

tile_linear_neural_02_basic.o: tile_linear_neural_02_basic.cpp
	$(COMPILER) $(COMPILEFLAGS) -c -o $@ $<

tile_nonlinear_gold.o: tile_nonlinear_gold.cpp
	$(COMPILER) $(COMPILEFLAGS) -c -o $@ $<

tile_nonlinear_neural_01_sucky.o: tile_nonlinear_neural_01_sucky.cpp
	$(COMPILER) $(COMPILEFLAGS) -c -o $@ $<

#########################

clean:
	@rm -fvr *.o $(EXENAME)

run: $(EXENAME)
	./$(EXENAME) $(ARGS)

valgrind: $(EXENAME)
	valgrind --leak-check=full ./$(EXENAME) $(ARGS)

