/// \file
/// File Name:                      main.cpp \n
/// Date created:                   Summer 2015 \n
/// Engineers:                      Conor Gardner \n
/// Special Compile Instructions:   --std=c++11 -lpng \n
/// Compiler:                       g++ \n
/// Target OS:                      Ubuntu Linux 16.04 \n
/// Target architecture:            x86 64-bit \n */

#include <png.h> // yucky standard png library
#include <cstdio>
#include <cstring>

#include "png.h" // my wrapper class

// Copyright stuff...
// 
// The read_from_file was adapted from the libpng examples in the libpng
// documentation and from the PNG class written by Chase Geigle for cs225 2012
// at the UofI

png_t::png_t()
{
    initialized = false;
    width = 0;
    height = 0;
}


png_t::png_t(const char* filename)
{
    initialized = false;
    read_from_file(filename);
    // read_from file will set all the member variables
}

        
png_t::png_t(bool greyscale, int width, int height)
{
    
    this->initialized = true;
    this->width = width;
    this->height = height;
    this->greyscale = greyscale;
    
    int pixels = width * height;
    
    if (greyscale)
    {
        this->red_grey = new unsigned char[pixels];
    } else {
        this->red_grey = new unsigned char[3 * pixels];
        this->green = red_grey + pixels;
        this->blue = red_grey + 2 * pixels;
    }
}


png_t::png_t(const png_t& rhs)
{
    
    // copy dimensions and flags
    initialized = rhs.initialized;
    greyscale = rhs.greyscale;
    width = rhs.width;
    height = rhs.height;
    
    // copy pixel data if it exists, making new arrays
    if (initialized)
    {
        
        int pixels = width * height;
        
        // at least this color channel contains data
        
        if (greyscale)
        {
            // only copy one channel
            red_grey = new unsigned char[pixels];
            memcpy(red_grey, rhs.red_grey, pixels * sizeof(char));
        } else {
            // copy all channels
            red_grey = new unsigned char[3 * pixels];
            green = red_grey + pixels;
            blue = red_grey + 2 * pixels;
            memcpy(red_grey, rhs.red_grey, 3 * pixels * sizeof(char));
        }
        
    }
    
}

png_t::~png_t()
{
    if (initialized)
        delete[] red_grey;
}

void png_t::clear()
{
    
    if (initialized)
    {
        delete[] red_grey;
        width = 0;
        height = 0;
        initialized = false;
    }
    
}

PNG_ERROR png_t::read_from_file(const char* filename)
{
    
    /* Yucky libpng initialization */
    
    // open file for reading in binary format
    FILE* file = fopen(filename, "rb");
    if (file == NULL)
    {
        fprintf(stderr, "Error (%s) failed to open file for reading\n", filename);
        return FILE_OPEN_FAILURE;
    }

    // read the 8-byte header of the file to make sure it's a png image
    unsigned char header[8];
    if (fread(header, 1, 8, file) < 1 || png_sig_cmp(header, 0, 8) != 0)
    {
        fprintf(stderr, "Error (%s) file is not a png\n", filename);
        fclose(file);
        return NOT_PNG;
    }
    
    // libpng uses 2 structures to keep track of the state of an open png file
    // these are the read struct and the info struct.  The next two function
    // calls dynamically allocate these structures
    png_structp read_struct = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (read_struct == NULL)
    {
        fprintf(stderr, "Error (%s) could not create png read struct\n", filename);
        fclose(file);
        return READ_STRUCT_FAILURE;
    }
    
    png_infop info_struct = png_create_info_struct(read_struct);
    if (info_struct == NULL)
    {
        fprintf(stderr, "Error (%s) could not create png info struct\n", filename);
        png_destroy_read_struct(&read_struct, NULL, NULL);
        fclose(file);
        return INFO_STRUCT_FAILURE;
    }
    
    // set error handling.  This registers a way to get back here if an error
    // occurrs.  Had we not set this, libpng would abort the whole program
    // if an error occurred.
    if (setjmp(png_jmpbuf(read_struct)) != 0)
    {
        fprintf(stderr, "Error (%s) failed to setup longjump 1 for libpng\n", filename);
        fclose(file);
        png_destroy_info_struct(read_struct, &info_struct);
        png_destroy_read_struct(&read_struct, NULL, NULL);
        return LIBPNG_CONFIG_FAILURE;
    }
    
    // initialize png for read operations
    png_init_io(read_struct, file);
    
    // let it know that we already read the first 8 bytes (for the header)
    png_set_sig_bytes(read_struct, 8);
    
    // initialize info struct
    png_read_info(read_struct, info_struct);
    
    // now that we've gotten past most of the error checking, delete previous
    // color data if it exists
    clear();
    
    width = png_get_image_width(read_struct, info_struct);
    height = png_get_image_height(read_struct, info_struct);
    int pixels = width * height;
    
    /* Hold on to your balls, because now we're *actually* gonna start
     * reading pixels from the image */
    
    // raw data will be read from the png in rows.  allocate a buffer for the rows
    int row_bytes = png_get_rowbytes(read_struct, info_struct);
    unsigned char* row_buff = new unsigned char[row_bytes];
    
    switch (png_get_color_type(read_struct, info_struct))
    {
        /* 256-bit color lookup table */
        case PNG_COLOR_TYPE_PALETTE:
        {
            
            greyscale = false;
            
            // retrieve the color palette - 256 full RGB colors
            png_color_struct* palette;
            int num_colors;
            png_get_PLTE(read_struct, info_struct, &palette, &num_colors);
            
            red_grey = new unsigned char[3 * pixels];
            green = red_grey + pixels;
            blue = red_grey + 2 * pixels;
            
            for (int iy = 0; iy < height; iy++)
            {
                
                // read a row of pixels using libpng
                png_read_row(read_struct, row_buff, NULL);
                
                // set up some base pointers to expedite indexing in the ix loop
                int iy_offset = iy * width;
                unsigned char* red_row = red_grey + iy_offset;
                unsigned char* green_row = green + iy_offset;
                unsigned char* blue_row = blue + iy_offset;
                
                // re-format the row loaded into the 3 member color channels
                for (int ix = 0; ix < width; ix++)
                {
                    png_color_struct* cur_color = palette + row_buff[ix];
                    red_row[ix] = cur_color->red;
                    green_row[ix] = cur_color->green;
                    blue_row[ix] = cur_color->blue;
                }
            }
            
            break;
            
        }
        
        /* Greyscale png */
        case PNG_COLOR_TYPE_GRAY:
        {
            
            greyscale = true;
            red_grey = new unsigned char[pixels];
            
            for (int iy = 0; iy < height; iy++)
            {
                // don't even need a buffer, memory layout is the same per-row
                png_read_row(read_struct, red_grey + iy * width, NULL);
            }            
            
            break;
        }
        
        /* Greyscale with transparency */
        case PNG_COLOR_TYPE_GRAY_ALPHA:
        {
            
            greyscale = true;
            red_grey = new unsigned char[pixels];
            
            for (int iy = 0; iy < height; iy++)
            {
                
                png_read_row(read_struct, row_buff, NULL);
                
                // set up some base pointers to expedite indexing in the ix loop
                unsigned char* grey_row = red_grey + iy * width;
                
                for(int ix = 0, iz = 0; ix < width; ix++, iz += 2)
                {
                    // skip the alpha channel
                    grey_row[ix] = row_buff[iz];
                }
                
            }
            
            break;
            
        }
        
        /* RGB */
        case PNG_COLOR_TYPE_RGB:
        {
            
            greyscale = false;
            // get the number of bytes per channel 
            int color_channels = png_get_channels(read_struct, info_struct);
            
            greyscale = false;
            
            red_grey = new unsigned char[3 * pixels];
            green = red_grey + pixels;
            blue = red_grey + 2 * pixels;
            
            for (int iy = 0; iy < height; iy++)
            {
                
                png_read_row(read_struct, row_buff, NULL);
                
                // set up some base pointers to expedite indexing in the ix loop
                unsigned char* row_cur = row_buff;
                int iy_offset = iy * width;
                unsigned char* red_row = red_grey + iy_offset;
                unsigned char* green_row = green + iy_offset;
                unsigned char* blue_row = blue + iy_offset;
                
                for (int ix = 0; ix < width; ix++)
                {
                    red_row[ix] = *row_cur;
                    green_row[ix] = *(row_cur + 1);
                    blue_row[ix] = *(row_cur + 2);
                    
                    // the RGB format may optionally be 3 or 4 bytes per pixel
                    // the fourth one would be a filler byte
                    row_cur += color_channels;
                }
                
            }
            
            break;
        }
        
        /* RGB with transparency */
        case PNG_COLOR_TYPE_RGB_ALPHA:
        {
            
            greyscale = false;
            
            red_grey = new unsigned char[3 * pixels];
            green = red_grey + pixels;
            blue = red_grey + 2 * pixels;
            
            // I'm not gonna do loop unrolling since g++ does it for me and
            // this function is already freakin-huge
            for (int iy = 0; iy < height; iy++)
            {
                
                // read a row of pixels using libpng
                png_read_row(read_struct, row_buff, NULL);
                
                // set up some base pointers to expedite indexing in the ix loop
                unsigned char* row_cur = row_buff;
                int iy_offset = iy * width;
                unsigned char* red_row = red_grey + iy_offset;
                unsigned char* green_row = green + iy_offset;
                unsigned char* blue_row = blue + iy_offset;
                
                // re-format the row loaded into the 3 member color channels
                for (int ix = 0; ix < width; ix++)
                {
                    red_row[ix] = *row_cur;
                    green_row[ix] = *(row_cur + 1);
                    blue_row[ix] = *(row_cur + 2);
                    // skip alpha channel
                    row_cur += 4;
                }
                
            }
            
            break;
            
        }
        default:
            fprintf(stderr, "Error (%s) unrecognised color type\n", filename);
            png_destroy_info_struct(read_struct, &info_struct);
            png_destroy_read_struct(&read_struct, NULL, NULL);
            fclose(file);
            return COLOR_FAILURE;
            break;
    }
    
    // cleanup
    delete[] row_buff;
    png_destroy_info_struct(read_struct, &info_struct);
    png_destroy_read_struct(&read_struct, NULL, NULL);
    fclose(file);
    
    initialized = true;
    return SUCCESS;
    
}

PNG_ERROR png_t::write_to_file(const char* filename) const
{
    
    /* Yucky libpng initialization */
    
    if (!initialized)
    {
        fprintf(stderr, "Error (%s) no image data to write\n", filename);
    }
    
    // open file for reading in binary format
    FILE* file = fopen(filename, "wb");
    if (file == NULL)
    {
        fprintf(stderr, "Error (%s) failed to open file for writing\n", filename);
        return FILE_OPEN_FAILURE;
    }
    
    // set error handling.  This registers a way to get back here if an error
    // occurrs.  Had we not set this, libpng would abort the whole program
    // if an error occurred.
    png_structp write_struct = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (write_struct == NULL)
    {
        fprintf(stderr, "Error (%s) could not create png write struct\n", filename);
        fclose(file);
        return WRITE_STRUCT_FAILURE;
    }
    
    png_infop info_struct = png_create_info_struct(write_struct);
    if (info_struct == NULL)
    {
        fprintf(stderr, "Error (%s) could not create png info struct\n", filename);
        png_destroy_write_struct(&write_struct, &info_struct);
        fclose(file);
        return INFO_STRUCT_FAILURE;
    }

    // set error handling.  This registers a way to get back here if an error
    // occurrs.  Had we not set this, libpng would abort the whole program
    // if an error occurred.
    if (setjmp(png_jmpbuf(write_struct)) != 0)
    {
        fprintf(stderr, "Error (%s) failed to setup longjump 2 for libpng\n", filename);
        fclose(file);
        png_destroy_info_struct(write_struct, &info_struct);
        png_destroy_write_struct(&write_struct, NULL);
        return LIBPNG_CONFIG_FAILURE;
    }
    
    png_init_io(write_struct, file);
    
    // write file header
    if (setjmp(png_jmpbuf(write_struct)) != 0)
	{
        fprintf(stderr, "Error (%s) failed to setup longjump 3 for libpng\n", filename);
        png_destroy_info_struct(write_struct, &info_struct);
        png_destroy_write_struct(&write_struct, NULL);
		fclose(file);
		return LIBPNG_CONFIG_FAILURE;
	}
    png_set_IHDR
    (
        write_struct,
        info_struct,
        width,
        height,
        8, // Bit depth
        greyscale ? PNG_COLOR_TYPE_GRAY : PNG_COLOR_TYPE_RGB,
        PNG_INTERLACE_NONE,
        PNG_COMPRESSION_TYPE_BASE,
        PNG_FILTER_TYPE_BASE
    );
    
    png_write_info(write_struct, info_struct);
    
    // write pixel data
    if (setjmp(png_jmpbuf(write_struct)) != 0)
	{
        fprintf(stderr, "Error (%s) failed to setup longjump 4 for libpng\n", filename);
        png_destroy_info_struct(write_struct, &info_struct);
		png_destroy_write_struct(&write_struct, NULL);
		fclose(file);
		return LIBPNG_CONFIG_FAILURE;
	}
    
    if (greyscale)
    {
        
        // 8-bit greyscale
        for (int iy = 0; iy < height; iy++)
        {
            // no conversions required - this classe's storage format happens
            // to match libpng's for 8-bit greyscale
            png_write_row(write_struct, red_grey + iy * width);
        }
        
    } else {
        
        int row_bytes = png_get_rowbytes(write_struct, info_struct);
        unsigned char* row_buff = new unsigned char[row_bytes];
        
        // 24-bit RGB image
        for (int iy = 0; iy < height; iy++)
        {
            
            // set up some base pointers to expedite indexing in the ix loop
            unsigned char* row_cur = row_buff;
            int iy_offset = iy * width;
            unsigned char* red_row = red_grey + iy_offset;
            unsigned char* green_row = green + iy_offset;
            unsigned char* blue_row = blue + iy_offset;
            
            for (int ix = 0; ix < width; ix++)
            {
                *row_cur = red_row[ix];
                *(row_cur + 1) = green_row[ix];
                *(row_cur + 2) = blue_row[ix];
                row_cur += 3;
            }

            png_write_row(write_struct, row_buff);
            
        }
        
        delete[] row_buff;
        
    }
    
    // cleanup
    png_write_end(write_struct, NULL);
    png_destroy_info_struct(write_struct, &info_struct);
    png_destroy_write_struct(&write_struct, NULL);
    fclose(file);
    
    return SUCCESS;
}


int png_t::get_width() const
{
    return width;
}

int png_t::get_height() const
{
    return height;
}

int png_t::get_pixels() const
{
    return width * height;
}

unsigned char* png_t::get_red() const
{
    if (initialized)
        return red_grey;
    else
        return NULL;
}

unsigned char* png_t::get_green() const
{
    if (initialized)
        if (greyscale)
            return red_grey;
        else
            return green;
    else
        return NULL;
}

unsigned char* png_t::get_blue() const
{
    if (initialized)
        if (greyscale)
            return red_grey;
        else
            return blue;
    else
        return NULL;
}

unsigned char* png_t::get_grey() const
{
    if (initialized && greyscale)
        return red_grey;
    else
        return NULL;
}

bool png_t::is_initialized() const
{
    return initialized;
}

bool png_t::is_grey() const
{
    return initialized && greyscale;
}

png_t& png_t::operator=(const png_t& rhs)
{
    
    // get rid of pre-existing data if it exists
    clear();
    
    // copy dimensions and flags
    initialized = rhs.initialized;
    greyscale = rhs.greyscale;
    width = rhs.width;
    height = rhs.height;
    
    // copy pixel data if it exists, making new arrays
    if (initialized)
    {
        
        int pixels = width * height;
        
        // at least this color channel contains data
        
        if (greyscale)
        {
            // only copy one channel
            red_grey = new unsigned char[pixels];
            memcpy(red_grey, rhs.red_grey, pixels * sizeof(char));
        } else {
            // copy all channels
            red_grey = new unsigned char[3 * pixels];
            green = red_grey + pixels;
            blue = red_grey + 2 * pixels;
            memcpy(red_grey, rhs.red_grey, 3 * pixels * sizeof(char));
        }
        
    }
        
    return *this;
}

bool png_t::operator==(const png_t& rhs) const
{
    
    // try to be lazy by comparing the dimensions first
    // ... if those are the same then check the red_grey channel
    // ... if that's still the same then check the green channel
    // ... and if all those match then finally check the blue channel
    
    if (greyscale != rhs.greyscale)
        return false;
    
    if (greyscale)
    {
        return
            (width == rhs.width && height == rhs.height)
         && (strcmp((const char*)(red_grey), (const char*)(rhs.red_grey)) == 0);
    } else {
        return
            (width == rhs.width && height == rhs.height)
         && (strcmp((const char*)(red_grey), (const char*)(rhs.red_grey)) == 0)
         && (strcmp((const char*)(green), (const char*)(rhs.green)) == 0)
         && (strcmp((const char*)(blue), (const char*)(rhs.blue)) == 0);
    }
    
     
}

bool png_t::operator!=(const png_t& rhs) const
{
    return !(*this == rhs);
}

