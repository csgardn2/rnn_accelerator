Generates a visualization of the heat maps and power dissipation maps generated
by the hotspot benchmark from the rodinia benchmark suite.  The program will try
to auto-detect the image dimensions or you can explictly specify them as the 
third and fouth parameters after the output png file name.

Typical command line usage:
    
    ./heat_visualizer temp_512 output.png
    ./heat_visualizer temp_512 output.png 512 512
    ./heat_visualizer power_256 output.png
    ./heat_visualizer power_256 output.png 256 256
    ./heat_visualizer output.out
    ./heat_visualizer output.out 128 128
    
