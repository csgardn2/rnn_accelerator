#!/usr/bin/python3.5

import math
import sys

# Open one or more files containing flattened 2D temperature maps or power maps
# (both share the same format) then load the contents of the file into a python
# array.
#
# Suppose your power maps are saved as temperature_00.csv, temperature_01.csv
# ... temperature_63.csv.  Then you would call this function as
# load_training_data("temperature_", ".csv", 0, 64, 256, 4).
# The return value will be a 3D array with dimensions array[64][4][256] and is
# indexed as array[time][y][x].
def load_training_data(     \
    filename_prefix_str,    \
    filename_suffix_str,    \
    first_index_uint,       \
    num_files_uint,         \
    width_uint,             \
    height_uint             \
):
    
    # Pre-allocate and zero initialize return value.  It can be quite large.
    # See function description for dimensions and layout
    ret_uint3d = num_files_uint * [height_uint * [width_uint * [0.0]]]
    
    # Iterate through each training data file
    full_index_len_uint = int(math.log(first_index_uint + num_files_uint - 1, 10)) + 1
    for ix in range(first_index_uint, first_index_uint + num_files_uint):
        
        # Compute the name of the current file to read
        index_str = str(ix)
        cur_filename_str =                                  \
            filename_prefix_str                             \
          + "0" * (full_index_len_uint - len(index_str))    \
          + index_str                                       \
          + filename_suffix_str
        
        # Open training data file for reading
        try:
            decoder_file = open(cur_filename_str, "r")
        except FileNotFoundError:
            print(                                                                          \
                "Error.  Training file \"{}\" does not exist.".format(cur_filename_str),    \
                file = sys.stderr                                                           \
            )
            continue
        except PermissionError:
            print(                                                                                          \
                "Error.  Insufficient permissions to open \"{}\" for reading.".format(cur_filename_str),    \
                file = sys.stderr                                                                           \
            )
            continue;
            
        # Begin reading the file.
        # The first line is the number of floats in the file
        # Damn you python, for not having something similar to >>
        
        
        print(cur_filename_str)
        

