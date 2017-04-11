COMPILER=nvcc
EXENAME=hotspot
COMPILEFLAGS=--compiler-options=-Wall,-Wextra,-Wfatal-errors -arch=sm_30 -ccbin=g++ -std=c++11
LINKFLAGS=--compiler-options=-Wall,-Wextra,-Wfatal-errors -arch=sm_30 -ccbin=g++
LIBS=
ARGS=256 1 256 ../../../heat_maps/power_and_initial_temperatures/random_temp_256.txt ../../../heat_maps/power_and_initial_temperatures/random_power_256.txt output.txt

all: $(EXENAME)

# add additional .o files on the line below (after main.o)
$(EXENAME): \
    dump_3x3_csv_training_data.o \
    dump_full_csv_training_data.o \
    dump_py_training_data.o \
    generate_power_map.o \
    hotspot.o \
    main.o \
    power_map_state.o
	$(COMPILER) $(LINKFLAGS) -o $(EXENAME) $^ $(LIBS)
	@echo "*** COMPILE_SUCCESSFUL ***"

#########################

# add more .cpp -> .o compile commands here

dump_3x3_csv_training_data.o: dump_3x3_csv_training_data.cpp
	$(COMPILER) $(COMPILEFLAGS) -c -o $@ $<

dump_full_csv_training_data.o: dump_full_csv_training_data.cpp
	$(COMPILER) $(COMPILEFLAGS) -c -o $@ $<

dump_py_training_data.o: dump_py_training_data.cpp
	$(COMPILER) $(COMPILEFLAGS) -c -o $@ $<

generate_power_map.o: generate_power_map.cpp
	$(COMPILER) $(COMPILEFLAGS) -c -o $@ $<

hotspot.o: hotspot.cpp
	$(COMPILER) $(COMPILEFLAGS) -c -o $@ $<

main.o: hotspot.cu
	$(COMPILER) $(COMPILEFLAGS) -c -o $@ $<

power_map_state.o: power_map_state.cpp
	$(COMPILER) $(COMPILEFLAGS) -c -o $@ $<

#########################

clean:
	@rm -fv *.o $(EXENAME)

run: $(EXENAME)
	./$(EXENAME) $(ARGS)

