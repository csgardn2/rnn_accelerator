
power_file = open('/home/cosine/research/rnn_accelerator/heat_maps/power_and_initial_temperatures/random_power_256.txt')
out_file = open('/home/cosine/research/rnn_accelerator/dnn/hotspot/data/input.txt', 'w')

power_list = [float(i) for i in power_file.readlines()]

out_file.write(str(5*65536) + '\n')
for i in xrange(5):
    temp_file = open('/home/cosine/research/rnn_accelerator/heat_maps/simulation_output_temperatures/256x256_random_iterations0000to0031/source_256x256_random_iteration000' + str(i) + '.csv')
    num_data = int(temp_file.readline())
    temp_list = [ [float(i) for i in temps.split(', ')]
            for temps in temp_file.readlines() ]
    for i in xrange(num_data):
        temp_list[i].append(power_list[i])
        out_file.write(str(temp_list[i]).strip('[]') + '\n')
    temp_file.close()

out_file.close()
power_file.close()
