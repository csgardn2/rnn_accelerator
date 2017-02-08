
f = open('/home/cosine/rnn_accelerator/heat_maps/simulation_output_temperatures/256x256_random_iterations0000to0031/source_256x256_random_iteration0000.csv')
num_pixels = int(f.readline())
input_image = [ [int(i) for i in inputs.split(', ')] for inputs in f.readlines()]

print num_pixels
