# you need to
#   sudo apt-get install parallel
# to run this script

DIM=256
ITERATIONS=256

> iteration_indexes.txt
> output_filenames.txt
> png_filenames.txt
for IX in $(seq 1 $ITERATIONS)
do
    echo "$IX" >> iteration_indexes.txt
    echo "./output_$IX.txt" >> output_filenames.txt
    echo "./output_$IX.png" >> png_filenames.txt
done

parallel --xapply -v ./hotspot ::: $DIM ::: 2 :::: iteration_indexes.txt ::: ../../data/hotspot/random_temp_$DIM.txt ::: ../../data/hotspot/random_power_$DIM.txt :::: output_filenames.txt
parallel --xapply -v ./heat_visualizer :::: output_filenames.txt :::: png_filenames.txt ::: 0 ::: 100

rm -f video.mp4
ffmpeg -framerate 4 -i './output_%d.png' -c:v libx264 -r 30 video.mp4

parallel -j 1 rm ::: -f :::: output_filenames.txt
parallel -j 1 rm ::: -f :::: png_filenames.txt
rm -f iteration_indexes.txt
rm -f output_filenames.txt
rm -f png_filenames.txt
