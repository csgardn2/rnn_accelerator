/// \file
/// File Name:                      args.cpp \n
/// Date created:                   Fri Feb 10 2017 \n
/// Engineers:                      Conor Gardner \n
/// Special Compile Instructions:   --std=c++1z \n
/// Compiler:                       g++ \n
/// Target OS:                      Ubuntu Linux 16.04 \n
/// Target architecture:            x86 64-bit \n */
/// See html/index.html or args.h for documentation.

// TODO Please change experimental/string_view to string_view when it is
// officially released
#include <experimental/string_view>
#include <climits>
#include <string>

#include "args.h"

/// \brief Scan through the elements in argv looking for a particular argument
/// \return The index into argv where either long_parameter or short_parameter
/// is found, whichever came first.  Returns UINT_MAX if neither the long nor
/// short version were found.
unsigned search_argv
(
    
    /// [in] Number of elements in argv
    unsigned argc,
    
    /// [in] Array of tokenzied arguments passed to \ref main
    char** argv,
    
    /// [in] Long version of a command line parameter (such as "--width").
    // TODO Please change std::experimental::string_view to std::string_view
    // when it's officially released.
    std::experimental::string_view long_parameter,
    
    /// [in] Short version of a command line parameter (such as "-w").
    // TODO Please change std::experimental::string_view to std::string_view
    // when it's officially released.
    std::experimental::string_view short_parameter
    
){
    
    for (unsigned ix = 0; ix < argc; ix++)
    {
        const char* cur_arg = argv[ix];
        if (long_parameter == cur_arg)
            return ix;
        if (short_parameter == cur_arg)
            return ix;
    }
    
    return UINT_MAX;
    
}

const char* arg_error_t::enum_to_string(arg_error_code_t error_code_)
{
    
    static const char* error_strings[unsigned(arg_error_code_t::NUM_ERROR_CODES)] = 
    {
        "Success",
        "Requied command line parameter --width / -w was not passed.",
        "Command line parameter --width / -w must be followed by an integer >= 0.",
        "Command line parameter --width / -w was followed by an invalid number.",
        "Requied command line parameter --height / -h was not passed.",
        "Command line parameter --height / -h must be followed by an integer >= 0.",
        "Command line parameter --height / -h was followed by an invalid number.",
        "Requied command line parameter --time-steps / -t was not passed.",
        "Command line parameter --time-steps / -t must be followed by an integer >= 0.",
        "Command line parameter --time-steps / -t was followed by an invalid number.",
        "Required command line parameter --base-filename / -o was not passed.",
        "Command line parameter --base-filename / -o must be followed by a string."
    };
    
    unsigned ix = unsigned(error_code_);
    if (ix < unsigned(arg_error_code_t::NUM_ERROR_CODES))
        return error_strings[ix];
    else
        return "";
    
}

arg_error_t args_t::parse(unsigned argc, char** argv, bool* consumed)
{
    
    // Mark all previous argument data as uninitialized
    this->width_status = arg_status_t::NOT_FOUND;
    this->height_status = arg_status_t::NOT_FOUND;
    this->time_steps_status = arg_status_t::NOT_FOUND;
    this->base_filename_status = arg_status_t::NOT_FOUND;
    
    bool enable_consumption_tracking = consumed != false;
    if (enable_consumption_tracking)
    {
        for (unsigned ix = 0; ix < argc; ix++)
            consumed[ix] = false;
    }
    
    // Parse width argument
    unsigned last_arg_ix = argc - 1;
    unsigned match = search_argv(argc, argv, "--width", "-w");
    if (match == UINT_MAX)
    {
        // This is a required argument that wasn't found.  Abort.
        this->width_status = arg_status_t::NOT_FOUND;
        return arg_error_code_t::WIDTH_NOT_PASSED;
    }
    if (enable_consumption_tracking)
        consumed[match] = true;
    if (match == last_arg_ix)
    {
        this->width_status = arg_status_t::INVALID;
        return arg_error_code_t::WIDTH_WITHOUT_ARGUMENT;
    }
    if (enable_consumption_tracking)
        consumed[match + 1] = true;
    signed converted;
    try
    {
        // Unfortunatly, we have to construct a string from argv[match] since
        // atoi causes undefined behavior if an invalid string is passed.
        converted = std::stoi(argv[match + 1]);
    } catch (std::exception error_code) {
        this->width_status = arg_status_t::INVALID;
        return arg_error_code_t::WIDTH_INVALID;
    }
    if (converted < 0)
    {
        this->width_status = arg_status_t::INVALID;
        return arg_error_code_t::WIDTH_INVALID;
    }
    // Conversion successful
    this->width = converted;
    this->width_status = arg_status_t::FOUND;
    
    // Parse height argument
    match = search_argv(argc, argv, "--height", "-h");
    if (match == UINT_MAX)
    {
        this->height_status = arg_status_t::NOT_FOUND;
        return arg_error_code_t::HEIGHT_NOT_PASSED;
    }
    if (enable_consumption_tracking)
        consumed[match] = true;
    if (match == last_arg_ix)
    {
        this->height_status = arg_status_t::INVALID;
        return arg_error_code_t::HEIGHT_WITHOUT_ARGUMENT;
    }
    if (enable_consumption_tracking)
        consumed[match + 1] = true;
    try
    {
        converted = std::stoi(argv[match + 1]);
    } catch (std::exception error_code) {
        this->height_status = arg_status_t::INVALID;
        return arg_error_code_t::HEIGHT_INVALID;
    }
    if (converted < 0)
    {
        this->height_status = arg_status_t::INVALID;
        return arg_error_code_t::HEIGHT_INVALID;
    }
    this->height = converted;
    this->height_status = arg_status_t::FOUND;
    
    // Parse time-steps argument
    match = search_argv(argc, argv, "--time-steps", "-t");
    if (match == UINT_MAX)
    {
        this->time_steps_status = arg_status_t::NOT_FOUND;
        return arg_error_code_t::TIME_STEPS_NOT_PASSED;
    }
    if (enable_consumption_tracking)
        consumed[match] = true;
    if (match == last_arg_ix)
    {
        this->time_steps_status = arg_status_t::INVALID;
        return arg_error_code_t::TIME_STEPS_WITHOUT_ARGUMENT;
    }
    if (enable_consumption_tracking)
        consumed[match + 1] = true;
    try
    {
        converted = std::stoi(argv[match + 1]);
    } catch (std::exception error_code) {
        this->time_steps_status = arg_status_t::INVALID;
        return arg_error_code_t::TIME_STEPS_INVALID;
    }
    if (converted < 0)
    {
        this->time_steps_status = arg_status_t::INVALID;
        return arg_error_code_t::TIME_STEPS_INVALID;
    }
    this->time_steps = converted;
    this->time_steps_status = arg_status_t::FOUND;
    
    // Parse base-filename argument
    match = search_argv(argc, argv, "--base-filename", "-o");
    if (match == UINT_MAX)
    {
        this->base_filename_status = arg_status_t::NOT_FOUND;
        return arg_error_code_t::BASE_FILENAME_NOT_PASSED;
    }
    if (enable_consumption_tracking)
        consumed[match] = true;
    if (match == last_arg_ix)
    {
        this->base_filename_status = arg_status_t::INVALID;
        return arg_error_code_t::BASE_FILENAME_WITHOUT_ARGUEMENT;
    }
    if (enable_consumption_tracking)
        consumed[match + 1] = true;
    this->base_filename = argv[match + 1];
    this->base_filename_status = arg_status_t::FOUND;
        
    // Sufficient argumetns parsed successfully
    return arg_error_code_t::SUCCESS;
    
}

