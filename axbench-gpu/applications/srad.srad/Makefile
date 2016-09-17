include ../config.mk

CUDA_DIR = /usr/local/cuda
CUDA_LIB_DIR := $(CUDA_DIR)/lib
CC := $(CUDA_DIR)/bin/nvcc
CFLAGS		:=
LFLAGS		:= -lParrot
HEADERS     := src
INCLUDE 	:= -I${ANALOG_INC} -I${FANN_INC} -I${HEADERS}
LIB			:= -L${ANALOG_LIB} -L${FANN_LIB} -L$(PARROT_LIB)

INCLUDE_BIN := $(CUDA_DIR)/include

# Example
# target: dependencies
	# command 1
	# command 2
          # .
          # .
          # .
	# command n

# link objects(binaries) together
a.out:	DIR ./obj/main.o
	$(CC)	./obj/main.o \
				-I$(INCLUDE_BIN) \
				-L$(CUDA_LIB_DIR)  \
				-lm -lcuda -lcudart -o ./bin/srad.out

DIR:
	@echo ${CPP_FILES}
	@echo ${OBJ_FILES}
	if [ ! -d "./bin" ];then 	\
		mkdir bin;				\
	fi
	if [ ! -d "./obj" ];then 	\
		mkdir obj;				\
	fi

# compile main function file into object (binary)
./obj/main.o: 	./src/main.cu \
				./src/define.c \
				./src/graphics.c \
				./src/extract_kernel.cu \
				./src/prepare_kernel.cu \
				./src/reduce_kernel.cu \
				./src/srad2_kernel.cu \
				./src/compress_kernel.cu
	nvcc	./src/main.cu \
				$(CFLAGS) $(INCLUDE) -I$(NVIDIA_HELPER) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -I$(PARROT_LIB) -c -O3 -arch sm_20 -o ./obj/main.o
	#python $(PLANG) -c $(PARROT_JSON) -a observe -e "$(CC) $(CFLAGS) $(INCLUDE) -I$(NVIDIA_HELPER) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -I$(PARROT_LIB) -O3 -arch sm_13 -c" -s "./src/main.cu" -o "$@" -t

# delete all object files
clean:
	rm ./bin/srad.out 

