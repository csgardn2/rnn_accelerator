for SIZE in 64 128 256 512 1024
do
    
    TEMPERATURE_NAME="random_temp_$SIZE.txt"
    > $TEMPERATURE_NAME
    POWER_NAME="random_power_$SIZE.txt"
    > $POWER_NAME
    
    echo "Generating $TEMPERATURE_NAME and $POWER_NAME"
    for IY in $(seq 1 $SIZE)
    do
        
        for IX in $(seq 1 $SIZE)
        do
            
            echo $((RANDOM % 100)) >> $TEMPERATURE_NAME
            echo $((RANDOM % 20 - 10))  >> $POWER_NAME
            
        done
        
    done
    
done

