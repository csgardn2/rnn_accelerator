See \ref args_t for command line usage

The code in this folder is designed to generate semi-random heat maps for the
hotspot benchmark from the rodinia benchmark suite by superimposing a bunch of
2D gaussians into the power map plane.  It is possible to directly call the
\ref generate_power_map function from within hotspot.cu (external) and feed
the generated power maps to the cuda kernel inside.  Alternativly, the code
in this folder may be compiled into a stand-alone utility which generates and
dumps heat maps into either txt or png files.

The hotspot.cu benchmark in tensorflow/hotspot/gold_code has already been
updated and power map generation has been integrated (but not tested).
