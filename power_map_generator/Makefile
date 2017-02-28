COMPILER=g++
EXENAME=power_map_generator
COMPILEFLAGS=-Wall -Wextra -ansi -Wfatal-errors -std=c++1z
LINKFLAGS=-Wall -Wextra -ansi -Wfatal-errors
LIBS=-lpng
ARGS=--width 256 --height 256 --time-steps 256 --base-filename power

all: documentation $(EXENAME)

# Add additional .o files on the line below (after main.o)
$(EXENAME): args.o hotspot.o main.o generate_power_map.o png.o power_map_state.o main.o
	$(COMPILER) $(LINKFLAGS) -o $(EXENAME) $^ $(LIBS)
	@echo "*** COMPILE_SUCCESSFUL ***"

#########################

# Add more .cpp -> .o compile commands here

args.o: args.cpp args.h
	$(COMPILER) $(COMPILEFLAGS) -c -o $@ $<

hotspot.o: hotspot.cpp hotspot.h
	$(COMPILER) $(COMPILEFLAGS) -c -o $@ $<

generate_power_map.o: generate_power_map.cpp generate_power_map.h
	$(COMPILER) $(COMPILEFLAGS) -c -o $@ $<

main.o: main.cpp args.h
	$(COMPILER) $(COMPILEFLAGS) -c -o $@ $<

png.o: png.cpp png.h
	$(COMPILER) $(COMPILEFLAGS) -c -o $@ $<

power_map_state.o: power_map_state.cpp power_map_state.h
	$(COMPILER) $(COMPILEFLAGS) -c -o $@ $<

Doxyfile:
	@echo "Couldn't find Doxyfile, generating default"
	doxygen -g

#########################

clean:
	@rm -fv *.o $(EXENAME)

run: $(EXENAME)
	./$(EXENAME) $(ARGS)

# Note that the only suppressions in stdlib.supp are those caused by including
# header files and were verified to be false positives.  Don't suppress any real
# errors!
valgrind: $(EXENAME)
	valgrind --suppressions=stdlib.supp --leak-check=full ./$(EXENAME) $(ARGS)

documentation: Doxyfile $(ls *.cpp *.h)
	doxygen Doxyfile

