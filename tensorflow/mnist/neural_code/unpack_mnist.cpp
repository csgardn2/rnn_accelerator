/// \file
/// File Name:                      unpack_mnsit.cpp \n
/// Date created:                   Fri Nov 4 2016 \n
/// Engineers:                      Conor Gardner \n
/// Special Compile Instructions:   --std=c++11 \n
/// Compiler:                       g++ \n
/// Target OS:                      Ubuntu Linux 14.04 \n
/// Target architecture:            x86 64-bit \n */

#include <cstdint>
#include <fstream>
#include <iostream>

int main()
{
    
    static const unsigned num_items = 16384;
    
    std::ifstream labels_in("dataset/mnist_train_labels.txt");
    
    std::ofstream labels_out("mnist_labels.py");
    
    labels_out
        << "# 1D list of "
        << num_items
        << " unsigned ints of the range [0 to 9]\n"
           "# corresponding to mnist_images\n"
           "mnist_labels = [";
    
    unsigned label;
    for (unsigned ix = 1; ix < num_items; ix++)
    {
        labels_in >> label;
        labels_out << label << ", ";
    }
    labels_in >> label;
    labels_out << label << "]\n";
    
    // Finished parsing labels
    
    std::ifstream images_in("dataset/mnist_train_images.txt");
    std::ofstream images_out("mnist_images.py");
    images_out
        << "# mnist_images[image_index][x]\n"
           "# 0 <= image_index < "
        << num_items
        << " \n"
           "# 0 <= x < 784 \n"
           "# List of "
        << num_items
        << " greyscale images.  Each image is 28x28 but\n"
           "# has been flattened to a row of 784 pixels\n"
           "# and each pixel value ranges from 0 (white) to 255 (black)\n"
           "mnist_images = [\n";
   
    for (unsigned image = 0; image < num_items; image++)
    {
        images_out << "    [";
        unsigned pixel;
        for (unsigned ix = 0; ix < 783; ix++)
        {
            images_in >> pixel;
            images_out << pixel << ", ";
        }
        images_in >> pixel;
        images_out << pixel << ']';
        if (image < num_items - 1)
            images_out << ',';
        images_out << '\n';
    }
    
    images_out << "]\n";
    
    // Finished parsing images
    
    std::cout << "Parse finished\n";
    
    return 0;
    
}

