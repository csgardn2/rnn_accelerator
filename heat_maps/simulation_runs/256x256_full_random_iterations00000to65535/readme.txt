1) These files were compressed using plzip (multithreaded, agressive compressor).
It looks like tar is smart enough to extract the compressed files without needing
the plzip tool though :)

    Example:
        
        tar -xf tar -xf 256x256_full_random_iterations01024to08191.tar.lz
        
    If this doesn't work:
        
        sudo apt-get install plzip
        plzip -d 256x256_full_random_iterations01024to08191.tar.lz
        tar -xf 256x256_full_random_iterations01024to08191.tar

2) Since full dataset is very large, it has been hosted on Google drive instead
of git.  You can download the full dataset using the following public link:

    https://drive.google.com/drive/folders/0B7HrVzZAPCbBUi1aT0paSm1EMm8?usp=sharing
