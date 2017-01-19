#ifndef PNG_HEADER_GUARD
#define PNG_HEADER_GUARD

enum PNG_ERROR
{
    SUCCESS,
    FILE_OPEN_FAILURE,
    NOT_PNG,
    READ_STRUCT_FAILURE,
    WRITE_STRUCT_FAILURE,
    INFO_STRUCT_FAILURE,
    LIBPNG_CONFIG_FAILURE,
    COLOR_FAILURE,
    DATA_READ_FAILURE
};

class png_t
{
    
    private:
        
        int width;
        /* Number of pixels across a row.
         * Zero for uninitialized png objects
         * One or more for initialized png objects */
        
        int height;
        /* Number of pixels down a column.
         * Zero for uninitialized png objects
         * One or more for initialized png objects */
        
        bool initialized;
        /* true
         *    - the current png object is valid and contains image data
         *    - the width and height members are greater than zerp
         *    - at least the red_grey array is initialized.  If greyscale
         *      is also true, then the green and blue arrays are also initialized
         * false
         *    - width and height are guaranteed to be zero
         *    - none of the red_grey, green, or blue arrays are valid.
         *      however, they are not necessarily set to NULL so you should
         *      check the initialized variable rather than doing a NULL check
         *      on one of the arrays. */
        
        bool greyscale;
        /* true
         *    - The image is in greyscale and therefore the red, green, and blue
         *      color channels are identical.  To save memory, only the
         *      red_grey array is allocated and contains the pixel data.
         *    - The green and blue arrays are not allocated but need not be
         *      NULL
         * false
         *    - The image is full color and therefore all 3 arrays (red_grey,
         *      green, and blue) are filled with color data if
         *      initialized == true. */  
        
        unsigned char* red_grey;
        /* Red channel or intensity if image is greyscale */
        
        unsigned char* green;
        /* Green channel */
        
        unsigned char* blue;
        /* Blue Channel */
        
    public:
        
        png_t();
        /* Construct an empty png object which returns NULL data */
        
        png_t(const char* filename);
        /* Construct a new png object an initialize it to a file.  Produces
         * the same result as calling the constructor png() followed by
         * read_from_file() but possibly a little faster */
        
        png_t(bool greyscale, int width, int height);
        /* Construct a new png object and allocate memory for the pixels
         * The memory will not be initialized */
        
        png_t(const png_t& rhs);
        /* Copy Constructor */
        
        ~png_t();
        /* Destructor */
        
        void clear();
        /* De-allocate pixel data.  Essentially like calling the destructor
         * and then calling the zero-parameter constructor.
         * Has no effect if the image is already empty. */
        
        PNG_ERROR read_from_file(const char* filename);
        /* Open the file at path 'filename' and initialize this png object
         * from the file.  Discards any pre-existing pixel data */
        
        PNG_ERROR write_to_file(const char* filename) const;
        /* Open the file at path 'filename' and stream out this png object
         * from the file.
         *
         * Unfortunetly, write_to_file only supports 2 modes: 24-bit RGB
         * and 8 bit-per-pixel greyscale. */
        
        int get_width() const;
        /* The number of pixels across the horizontal dimension of the image
         * returns zero for an uninitialized png */
        
        int get_height() const;
        /* The number of pixels down the vertical dimension of the image
         * returns zero for an uninitialized png */
        
        int get_pixels() const;
        /* Returns the number of pixels in the image, which is the same as
         * width() * height() */
        
        unsigned char* get_red() const;
        unsigned char* get_green() const;
        unsigned char* get_blue() const;
        /* Get pointers to the color channel arrays.
         * You may safely read and write data to these pixels but do not
         * del-allocate, re-allocate, or resize the arrays since they
         * are internally used by the class (yes i know this breaks
         * encapsulation but do you really want to waste time copying big
         * arrays over and over again... :P.  If you really want to make a
         * copy, use operator=() or the copy constructor 
         *
         * The location of a pixel within any of the return arrays can be
         * calculated as follows:
         *      unsigned char red_pixel = get_red() + get_width() * y + x;
         *
         * There is no padding at the end of the rows and the arrays are
         * guarenteed to be contiguous */
        
        unsigned char* get_grey() const;
        /* See above.  If the image is greyscale, then return a pointer to
         * the pixel array.  If the image is RGB or not initialized, then
         * this function returns NULL.  Use the is_greyscale() function to
         * check if the image is greyscale */
        
        bool is_initialized() const;
        /* Returns true if the png object contains image data
         * Returns false if the png dosn't hold pixel data (the object
         * itself is still valid and can be initialized later)*/
        
        bool is_grey() const;
        /* Returns true if an image is greyscale (one color channel).
         * Returns false if an image is full color RGB
         * Output is undefined for uninitialized images */
        
        png_t& operator=(const png_t& rhs);
        /* Assignment (copy) */
        
        bool operator==(const png_t& rhs) const;
        /* Returns true if *this and rhs are pixel-for-pixel identical */
        
        bool operator!=(const png_t& rhs) const;
        /* Inverse of operator== */
        
};

#endif
