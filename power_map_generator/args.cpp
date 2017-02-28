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

const char* arg_error_t::enum_to_string(arg_error_code_t error_code_)
{
    
    static const char* error_strings[unsigned(arg_error_code_t::NUM_ERROR_CODES)] = 
    {
        "Success",
        "Requied command line parameter --width / -w was not passed.",
        "Command line parameter --width / -w was followed by an invalid number.",
        "Requied command line parameter --height / -h was not passed.",
        "Command line parameter --height / -h was followed by an invalid number.",
        "Requied command line parameter --time-steps / -t was not passed.",
        "Command line parameter --time-steps / -t was followed by an invalid number.",
        "Required command line parameter --base-filename / -o was not passed.",
        "Command line parameter --base-filename / -o must be followed by a string.",
    };
    
    unsigned ix = unsigned(error_code_);
    if (ix < unsigned(arg_error_code_t::NUM_ERROR_CODES))
        return error_strings[ix];
    else
        return "";
    
}

arg_error_t args_t::parse(unsigned argc, char const* const* argv, bool* consumed)
{
    
    // Mark all previous argument data as uninitialized
    this->width_status = arg_status_t::NOT_FOUND;
    this->height_status = arg_status_t::NOT_FOUND;
    this->time_steps_status = arg_status_t::NOT_FOUND;
    this->base_filename_status = arg_status_t::NOT_FOUND;
    
    bool enable_consumption_tracking = consumed != nullptr;
    if (enable_consumption_tracking)
    {
        for (unsigned ix = 0; ix < argc; ix++)
            consumed[ix] = false;
    }
    
    // --width -w
    switch (args_t::parse_unsigned_argument(
        argc, argv, "--width", "-w", &(this->width_status), &(this->width), consumed
    )){
        case arg_status_t::NOT_FOUND:
            return arg_error_code_t::WIDTH_NOT_PASSED;
        case arg_status_t::INVALID:
            return arg_error_code_t::WIDTH_INVALID;
        default:
            break;
    }
    
    // --height -h
    switch (args_t::parse_unsigned_argument(
        argc, argv, "--height", "-h", &(this->height_status), &(this->height), consumed
    )){
        case arg_status_t::NOT_FOUND:
            return arg_error_code_t::HEIGHT_NOT_PASSED;
        case arg_status_t::INVALID:
            return arg_error_code_t::WIDTH_INVALID;
        default:
            break;
    }
    
    // --time-steps -t
    switch (args_t::parse_unsigned_argument(
        argc, argv, "--time-steps", "-t", &(this->time_steps_status), &(this->time_steps), consumed
    )){
        case arg_status_t::NOT_FOUND:
            return arg_error_code_t::TIME_STEPS_NOT_PASSED;
        case arg_status_t::INVALID:
            return arg_error_code_t::TIME_STEPS_INVALID;
        default:
            break;
    }
    
    // --base-filename -o
    unsigned match = search_argv(argc, argv, "--base-filename", "-o");
    if (match == UINT_MAX)
    {
        this->base_filename_status = arg_status_t::NOT_FOUND;
        return arg_error_code_t::BASE_FILENAME_NOT_PASSED;
    }
    if (enable_consumption_tracking)
        consumed[match] = true;
    if (match + 1 >= argc)
    {
        this->base_filename_status = arg_status_t::INVALID;
        return arg_error_code_t::BASE_FILENAME_INVALID;
    }
    if (enable_consumption_tracking)
        consumed[match + 1] = true;
    this->base_filename = argv[match + 1];
    this->base_filename_status = arg_status_t::FOUND;
        
    // Sufficient arguments parsed successfully
    return arg_error_code_t::SUCCESS;
    
}

unsigned args_t::search_argv
(
    unsigned argc,
    char const* const* argv,
    std::experimental::string_view long_parameter,
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

arg_status_t args_t::parse_unsigned_argument
(
    unsigned argc,
    char const* const* argv,
    std::experimental::string_view long_parameter,
    std::experimental::string_view short_parameter,
    arg_status_t* arg_status,
    unsigned* parameter_value,
    bool* consumed
){
    
    bool enable_consumption_tracking = consumed != nullptr;
    
    // Find the first part of the command line argument,
    // the part starting with -- or -
    unsigned match = args_t::search_argv(argc, argv, long_parameter, short_parameter);
    if (match == UINT_MAX)
    {
        *arg_status = arg_status_t::NOT_FOUND;
        return arg_status_t::NOT_FOUND;
    }
    if (enable_consumption_tracking)
        consumed[match] = true;
    
    // Verify that there can be a value after the argument
    if (match + 1 >= argc)
    {
        *arg_status = arg_status_t::INVALID;
        return arg_status_t::INVALID;
    }
    if (enable_consumption_tracking)
        consumed[match + 1] = true;
    
    // Begin parsing the value after the command line switch
    signed converted;
    try
    {
        // Unfortunatly, we have to construct a string from argv[match] since
        // atoi causes undefined behavior if an invalid string is passed.
        converted = std::stoi(argv[match + 1]);
    } catch (std::exception error_code) {
        *arg_status = arg_status_t::INVALID;
        return arg_status_t::INVALID;
    }
    if (converted < 0)
    {
        *arg_status = arg_status_t::INVALID;
        return arg_status_t::INVALID;
    }
    
    // All argument validation checks passed.  Write back parameter value.
    *parameter_value = unsigned(converted);
    *arg_status = arg_status_t::FOUND;
    return arg_status_t::FOUND;
    
}

arg_status_t args_t::parse_float_argument
(
    unsigned argc,
    char const* const* argv,
    std::experimental::string_view long_parameter,
    std::experimental::string_view short_parameter,
    arg_status_t* arg_status,
    float* parameter_value,
    bool* consumed
){
    
    bool enable_consumption_tracking = consumed != nullptr;
    
    // Find the first part of the command line argument,
    // the part starting with -- or -
    unsigned match = args_t::search_argv(argc, argv, long_parameter, short_parameter);
    if (match == UINT_MAX)
    {
        *arg_status = arg_status_t::NOT_FOUND;
        return arg_status_t::NOT_FOUND;
    }
    if (enable_consumption_tracking)
        consumed[match] = true;
    
    // Verify that there can be a value after the argument
    if (match + 1 >= argc)
    {
        *arg_status = arg_status_t::INVALID;
        return arg_status_t::INVALID;
    }
    if (enable_consumption_tracking)
        consumed[match + 1] = true;
    
    // Begin parsing the value after the command line switch
    float converted;
    try
    {
        // Unfortunatly, we have to construct a string from argv[match] since
        // atoi causes undefined behavior if an invalid string is passed.
        converted = std::stof(argv[match + 1]);
    } catch (std::exception error_code) {
        *arg_status = arg_status_t::INVALID;
        return arg_status_t::INVALID;
    }
    if (converted < 0)
    {
        *arg_status = arg_status_t::INVALID;
        return arg_status_t::INVALID;
    }
    
    // All argument validation checks passed.  Write back parameter value.
    *parameter_value = converted;
    *arg_status = arg_status_t::FOUND;
    return arg_status_t::FOUND;
    
}

