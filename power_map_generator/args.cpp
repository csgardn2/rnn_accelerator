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

const char* args_t::enum_to_string(parsing_status_t parsing_status)
{
    
    static const char* error_strings[unsigned(parsing_status_t::NUM_ERROR_CODES)] = 
    {
        "Success",
        
        "Requied command line parameter --width / -w was not passed.",
        "Command line parameter --width / -w was followed by an invalid number.",
        "Requied command line parameter --height / -h was not passed.",
        "Command line parameter --height / -h was followed by an invalid number.",
        "Requied command line parameter --time-steps / -t was not passed.",
        "Command line parameter --time-steps / -t was followed by an invalid number.",
        
        "You must specify at least one of --base-png-filename -p and/or --base-txt-filename / -x.",
        "Command line parameter --base-png-filename / -p must be followed by a string.",
        "Command line parameter --base-txt-filename / -x must be followed by a string.",
        
        "Command line parameter --max-hotspots / -h was followed by an invalid number.",
        
        "Command line parameter --min-peak-amplitude / -a was followed by an invalid number.",
        "Command line parameter --min-peak-amplitude / -a was greater than the default max peak amplitude.",
        "Command line parameter --max-peak-amplitude / -A was followed by an invalid number.",
        "Command line parameter --max-peak-amplitude / -A was less than the default min peak amplitude.",
        "Command line parameter --min-peak-amplitude / -a was greater than --max-peak-amplitude / -A.",
        
        "Command line parameter --min-stddev / -s was followed by an invalid number.",
        "Command line parameter --min-stddev / -a was greater than the default max stddev.",
        "Command line parameter --max-stddev / -S was followed by an invalid number.",
        "Command line parameter --max-stddev / -a was less than the default min stddev.",
        "Command line parameter --min-stddev / -s was greater than --max-stddev / -S.",
        
        "Command line parameter --min-aging-rate / -r was followed by an invalid number.",
        "Command line parameter --min-aging-rate / -r was greater than the default max aging rate.",
        "Command line parameter --max-aging-rate / -R was followed by an invalid number.",
        "Command line parameter --max-aging-rate / -R was less than the default min aging rate.",
        "Command line parameter --min-aging-rate / -r was greater than --max-aging-rate / -R."
        
    };
    
    unsigned ix = unsigned(parsing_status);
    if (ix < unsigned(parsing_status_t::NUM_ERROR_CODES))
        return error_strings[ix];
    else
        return "";
    
}

parsing_status_t args_t::parse(unsigned argc, char const* const* argv, bool* consumed)
{
    
    // Mark all previous argument data as uninitialized incase a parsing
    // failure causes an early return.
    this->width_status = arg_status_t::NOT_FOUND;
    this->height_status = arg_status_t::NOT_FOUND;
    this->time_steps_status = arg_status_t::NOT_FOUND;
    this->base_png_filename_status = arg_status_t::NOT_FOUND;
    this->base_txt_filename_status = arg_status_t::NOT_FOUND;
    this->max_hotspots_status = arg_status_t::NOT_FOUND;
    this->min_peak_amplitude_status = arg_status_t::NOT_FOUND;
    this->max_peak_amplitude_status = arg_status_t::NOT_FOUND;
    this->min_stddev_status = arg_status_t::NOT_FOUND;
    this->max_stddev_status = arg_status_t::NOT_FOUND;
    this->min_aging_rate_status = arg_status_t::NOT_FOUND;
    this->max_aging_rate_status = arg_status_t::NOT_FOUND;
    
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
            return parsing_status_t::WIDTH_NOT_PASSED;
        case arg_status_t::INVALID:
            return parsing_status_t::WIDTH_INVALID;
        default:
            break;
    }
    
    // --height -h
    switch (args_t::parse_unsigned_argument(
        argc, argv, "--height", "-h", &(this->height_status), &(this->height), consumed
    )){
        case arg_status_t::NOT_FOUND:
            return parsing_status_t::HEIGHT_NOT_PASSED;
        case arg_status_t::INVALID:
            return parsing_status_t::WIDTH_INVALID;
        default:
            break;
    }
    
    // --time-steps -t
    switch (args_t::parse_unsigned_argument(
        argc, argv, "--time-steps", "-t", &(this->time_steps_status), &(this->time_steps), consumed
    )){
        case arg_status_t::NOT_FOUND:
            return parsing_status_t::TIME_STEPS_NOT_PASSED;
        case arg_status_t::INVALID:
            return parsing_status_t::TIME_STEPS_INVALID;
        default:
            break;
    }
    
    // --base-png-filename -p
    unsigned match = search_argv(argc, argv, "--base-png-filename", "-p");
    if (match == UINT_MAX)
    {
        
        this->base_png_filename_status = arg_status_t::NOT_FOUND;
        
    } else {
        
        if (enable_consumption_tracking)
            consumed[match] = true;
        
        if (match + 1 >= argc)
        {
            this->base_png_filename_status = arg_status_t::INVALID;
            return parsing_status_t::BASE_PNG_FILENAME_INVALID;
        }
        if (enable_consumption_tracking)
            consumed[match + 1] = true;
        this->base_png_filename = argv[match + 1];
        this->base_png_filename_status = arg_status_t::FOUND;
        
    }
    
    // --base-txt-filename -x
    match = search_argv(argc, argv, "--base-txt-filename", "-x");
    if (match == UINT_MAX)
    {
        
        this->base_txt_filename_status = arg_status_t::NOT_FOUND;
        
    } else {
        
        if (enable_consumption_tracking)
            consumed[match] = true;
        
        if (match + 1 >= argc)
        {
            this->base_txt_filename_status = arg_status_t::INVALID;
            return parsing_status_t::BASE_TXT_FILENAME_INVALID;
        }
        if (enable_consumption_tracking)
            consumed[match + 1] = true;
        this->base_txt_filename = argv[match + 1];
        this->base_txt_filename_status = arg_status_t::FOUND;
        
    }
    
    if (
        this->base_png_filename_status != arg_status_t::FOUND
     && this->base_txt_filename_status != arg_status_t::FOUND
    )
        return parsing_status_t::BASE_FILENAME_NOT_PASSED;
    
    // --max-hotspots -m
    switch (args_t::parse_unsigned_argument(
        argc, argv, "--max-hotspots", "-h", &(this->max_hotspots_status), &(this->max_hotspots), consumed
    )){
        case arg_status_t::NOT_FOUND:
            this->max_hotspots = args_t::default_max_hotspots;
            this->max_hotspots_status = arg_status_t::DEFAULTED;
            break;
        case arg_status_t::INVALID:
            return parsing_status_t::MAX_HOTSPOTS_INVALID;
        default:
            break;
    }
    
    // --min-peak-amplitude -a
    switch (args_t::parse_float_argument(
        argc, argv, "--min-peak-amplitude", "-a", &(this->min_peak_amplitude_status), &(this->min_peak_amplitude), consumed
    )){
        case arg_status_t::NOT_FOUND:
            this->min_peak_amplitude = args_t::default_min_peak_amplitude;
            this->min_peak_amplitude_status = arg_status_t::DEFAULTED;
            break;
        case arg_status_t::INVALID:
            return parsing_status_t::MIN_PEAK_AMPLITUDE_INVALID;
        default:
            break;
    }
    
    // --max-peak-amplitude -A TODO
    switch (args_t::parse_float_argument(
        argc, argv, "--max-peak-amplitude", "-A", &(this->max_peak_amplitude_status), &(this->max_peak_amplitude), consumed
    )){
        case arg_status_t::NOT_FOUND:
            this->max_peak_amplitude = args_t::default_max_peak_amplitude;
            this->max_peak_amplitude_status = arg_status_t::DEFAULTED;
            break;
        case arg_status_t::INVALID:
            return parsing_status_t::MAX_PEAK_AMPLITUDE_INVALID;
        default:
            break;
    }
    
    if (this->min_peak_amplitude > this->max_peak_amplitude)
    {
        if (this->min_peak_amplitude_status == arg_status_t::DEFAULTED)
        {
            this->max_peak_amplitude_status = arg_status_t::INVALID;
            return parsing_status_t::MAX_PEAK_AMPLITUDE_CONFLICTS_WITH_DEFAULT;
        } else if (this->max_peak_amplitude_status == arg_status_t::DEFAULTED) {
            this->min_peak_amplitude_status = arg_status_t::INVALID;
            return parsing_status_t::MIN_PEAK_AMPLITUDE_CONFLICTS_WITH_DEFAULT;
        } else {
            return parsing_status_t::PEAK_AMPLITUDE_CONFLICT;
        }
    }
    
    // --min-stddev -s
    switch (args_t::parse_float_argument(
        argc, argv, "--min-stddev", "-s", &(this->min_stddev_status), &(this->min_stddev), consumed
    )){
        case arg_status_t::NOT_FOUND:
            this->min_stddev = args_t::default_min_stddev;
            this->min_stddev_status = arg_status_t::DEFAULTED;
            break;
        case arg_status_t::INVALID:
            return parsing_status_t::MIN_STDDEV_INVALID;
        default:
            break;
    }
    
    // --max-stddev -S
    switch (args_t::parse_float_argument(
        argc, argv, "--max-stddev", "-S", &(this->max_stddev_status), &(this->max_stddev), consumed
    )){
        case arg_status_t::NOT_FOUND:
            this->max_stddev = args_t::default_max_stddev;
            this->max_stddev_status = arg_status_t::DEFAULTED;
            break;
        case arg_status_t::INVALID:
            return parsing_status_t::MAX_STDDEV_INVALID;
        default:
            break;
    }
    
    if (this->min_stddev > this->max_stddev)
    {
        if (this->min_stddev_status == arg_status_t::DEFAULTED)
        {
            this->max_stddev_status = arg_status_t::INVALID;
            return parsing_status_t::MAX_STDDEV_CONFLICTS_WITH_DEFAULT;
        } else if (this->max_stddev_status == arg_status_t::DEFAULTED) {
            this->min_stddev_status = arg_status_t::INVALID;
            return parsing_status_t::MIN_STDDEV_CONFLICTS_WITH_DEFAULT;
        } else {
            return parsing_status_t::STDDEV_CONFLICT;
        }
    }
    
    // --min-aging-rate -r
    switch (args_t::parse_float_argument(
        argc, argv, "--min-aging-rate", "-a", &(this->min_aging_rate_status), &(this->min_aging_rate), consumed
    )){
        case arg_status_t::NOT_FOUND:
            this->min_aging_rate = args_t::default_min_aging_rate;
            this->min_aging_rate_status = arg_status_t::DEFAULTED;
            break;
        case arg_status_t::INVALID:
            return parsing_status_t::MIN_AGING_RATE_INVALID;
        default:
            break;
    }
    
    // --max-aging-rate -R
    switch (args_t::parse_float_argument(
        argc, argv, "--max-aging-rate", "-A", &(this->max_aging_rate_status), &(this->max_aging_rate), consumed
    )){
        case arg_status_t::NOT_FOUND:
            this->max_aging_rate = args_t::default_max_aging_rate;
            this->max_aging_rate_status = arg_status_t::DEFAULTED;
            break;
        case arg_status_t::INVALID:
            return parsing_status_t::MAX_AGING_RATE_INVALID;
        default:
            break;
    }
    
    if (this->min_aging_rate > this->max_aging_rate)
    {
        if (this->min_aging_rate_status == arg_status_t::DEFAULTED)
        {
            this->max_aging_rate_status = arg_status_t::INVALID;
            return parsing_status_t::MAX_AGING_RATE_CONFLICTS_WITH_DEFAULT;
        } else if (this->max_aging_rate_status == arg_status_t::DEFAULTED) {
            this->min_aging_rate_status = arg_status_t::INVALID;
            return parsing_status_t::MIN_AGING_RATE_CONFLICTS_WITH_DEFAULT;
        } else {
            return parsing_status_t::AGING_RATE_CONFLICT;
        }
    }
    
    // Sufficient arguments parsed successfully
    return parsing_status_t::SUCCESS;
    
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
    
    // All argument validation checks passed.  Write back parameter value.
    *parameter_value = converted;
    *arg_status = arg_status_t::FOUND;
    return arg_status_t::FOUND;
    
}

