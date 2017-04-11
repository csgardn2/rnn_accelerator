#!/bin/bash

for PYRAMID_HEIGHT in {1..16}
do
    echo "*** Pyramid Height = ${PYRAMID_HEIGHT} ***"
    for TRIAL in {1..5}
    do
        echo "* Trial = ${TRIAL} *"
        time ./hotspot 256 ${PYRAMID_HEIGHT} 1000000 ../../../heat_maps/initial_conditions/random_temp_256.txt ../../../heat_maps/initial_conditions/random_power_256.txt output.txt > /dev/null
        echo
    done
done
