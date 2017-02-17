out_file = open('/home/cosine/research/rnn_accelerator/dnn/hotspot/data/golden.txt', 'w')

out_file.write(str(5*65536) + '\n')
for i in xrange(5):
    in_file = open('/home/cosine/research/rnn_accelerator/heat_maps/simulation_output_temperatures/256x256_random_iterations0000to0031/destination_256x256_random_iteration000' + str(i) + '.csv')
    in_file.readline()
    out_file.write(in_file.read())
    in_file.close()

out_file.close()
