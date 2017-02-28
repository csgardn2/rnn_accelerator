/// \file
/// File Name:                      args.h \n
/// Date created:                   Fri Feb 10 2017 \n
/// Engineers:                      Conor Gardner \n
/// Special Compile Instructions:   --std=c++11 \n
/// Compiler:                       g++ \n
/// Target OS:                      Ubuntu Linux 16.04 \n
/// Target architecture:            x86 64-bit \n */
/// See html/index.html for documentation.
 
#ifndef HEADER_GUARD_ARGS
#define HEADER_GUARD_ARGS

#include <string>
#include <experimental/string_view>

/// \brief Used to indicate weather an parameter was passed via the command line
/// or not.
enum class arg_status_t : unsigned char
{
    /// \brief The relevant command was neither passed on the command line nor
    /// does it have a default value.  The corresponding variable in args_t
    /// is uninitialized.
    NOT_FOUND,
    
    /// \brief The relevant command line parameter was explicitly passed on the
    /// command line and successfully parsed.
    FOUND,
    
    /// \brief The relevant command line parameter was not passed on the command
    /// line so the corresponding variable in \ref args_t was initialized to a
    /// default value.
    DEFAULTED,
    
    /// \brief The user attempted to explicitly pass the relevant command line
    /// parameter but the parameter was invalid and could not be parsed.
    INVALID
};

/// \brief Integral tag for an error which could occur within \ref args_t::parse
/// "parse" for easy comparison.
enum class parsing_status_t : unsigned char
{
    
    SUCCESS,
    
    WIDTH_NOT_PASSED,
    WIDTH_INVALID,
    HEIGHT_NOT_PASSED,
    HEIGHT_INVALID,
    TIME_STEPS_NOT_PASSED,
    TIME_STEPS_INVALID,
    
    BASE_FILENAME_NOT_PASSED,
    BASE_FILENAME_INVALID,
    
    MAX_HOTSPOTS_INVALID,
    
    MIN_PEAK_AMPLITUDE_INVALID,
    MIN_PEAK_AMPLITUDE_CONFLICTS_WITH_DEFAULT,
    MAX_PEAK_AMPLITUDE_INVALID,
    MAX_PEAK_AMPLITUDE_CONFLICTS_WITH_DEFAULT,
    PEAK_AMPLITUDE_CONFLICT,
    
    MIN_STDDEV_INVALID,
    MIN_STDDEV_CONFLICTS_WITH_DEFAULT,
    MAX_STDDEV_INVALID,
    MAX_STDDEV_CONFLICTS_WITH_DEFAULT,
    STDDEV_CONFLICT,
    
    MIN_AGING_RATE_INVALID,
    MIN_AGING_RATE_CONFLICTS_WITH_DEFAULT,
    MAX_AGING_RATE_INVALID,
    MAX_AGING_RATE_CONFLICTS_WITH_DEFAULT,
    AGING_RATE_CONFLICT,
    
    /// \brief This element MUST be the last enum.
    NUM_ERROR_CODES
    
};

/// \brief When the command line parameters to \ref main are parsed, they are
/// collected here for easy access.
class args_t
{
    
    public:
        
        /// \brief Construct a skeliton args_t object with all of the parameter
        ///variables flagged as uninitialized.
        inline args_t()
          : width_status(arg_status_t::NOT_FOUND),
            height_status(arg_status_t::NOT_FOUND),
            time_steps_status(arg_status_t::NOT_FOUND),
            base_filename_status(arg_status_t::NOT_FOUND),
            max_hotspots_status(arg_status_t::NOT_FOUND),
            min_peak_amplitude_status(arg_status_t::NOT_FOUND),
            max_peak_amplitude_status(arg_status_t::NOT_FOUND),
            min_stddev_status(arg_status_t::NOT_FOUND),
            max_stddev_status(arg_status_t::NOT_FOUND),
            min_aging_rate_status(arg_status_t::NOT_FOUND),
            max_aging_rate_status(arg_status_t::NOT_FOUND)
        {
            // Intentionally left blank
        }
        
        /// \brief See \ref parse
        inline args_t(unsigned argc, char const* const* argv, bool* consumed = nullptr)
        {
            this->parse(argc, argv, consumed);
        }
        
        /// \brief Parse the command line parameters passed to \ref main and
        /// initialize all relevant args_t variables if the corresponding
        /// parameter is passed.
        /// \return \ref parsing_status_t "SUCCESS" if sufficient parameters
        /// were successfully parsed or defaulted to execute the rest of the
        /// program.
        parsing_status_t parse
        (
            
            /// [in] The same as argc from \ref main.  The number of elements
            /// in the argv array and the optional consumed array.
            unsigned argc,
            /// [in] Array of strings passed from \ref main.  Each is a token
            /// passed as a command line parameter.
            char const* const* argv,
            
            /// [out].  Optional.  If you want to verify which command line
            /// parameters were recognized and consumed, allocate an pass an
            /// array of bools the same size as argv here.  If a command line
            /// parameter (and possibly its argument) were consumed, their
            /// corresponding element in the consumed array is set to true.
            /// Note that elements are set to true even if an argument was
            /// invalid.
            bool* consumed = nullptr
            
        );
        
        /// \brief Create a human-readable error message suitable for printing.
        static const char* enum_to_string(parsing_status_t parsing_status);
        
        /// \brief Required.\n
        /// Usage: --width -w [integer >= 0]\n
        /// The number of columns in the output power map.
        unsigned width;
        
        /// See \ref width and arg_status_t
        arg_status_t width_status;
        
        /// \brief Required.\n
        /// Usage: --height -h [integer >= 0]\n
        /// The number of rows in the output power map.
        unsigned height;
        
        /// See \ref height and arg_status_t
        arg_status_t height_status;
        
        /// \brief Required.\n
        /// Usage: --time-steps -t [integer >= 0]\n
        /// The number output power maps (files) to generate.
        unsigned time_steps;
        
        /// See \ref time_steps and arg_status_t
        arg_status_t time_steps_status;
        
        /// \brief Required.\n
        /// Usage: --base-filename -o [string]\n
        /// The base part of a filename which will be used to save power
        /// maps.
        /// Example:
        /// The user passes '--base-filename lard' and '--time-steps 4' on the
        /// command line.  The output files will be named 'lard_0.txt',
        /// 'lard_1.txt', 'lard_2.txt', and 'lard_3.txt'.
        std::string base_filename;
        
        /// See \ref base_filename and arg_status_t
        arg_status_t base_filename_status;
        
        /// \brief Optional\n
        /// Usage: --max-hotspots -h [integer >= 0]\n
        /// See \ref args_t::default_max_hotspots "default value" and \ref
        /// power_map_state_t::max_hotspots
        unsigned max_hotspots;
        
        /// See \ref power_map_state_t::max_hotspots and arg_status_t
        arg_status_t max_hotspots_status;
        
        /// \brief If the associated argument is not passed on the command line,
        /// this value is assumed.  Note that passing and \ref arg_status_t
        /// "INVALID" or mal-formed argument still generates an error and the
        /// default value will not be assumed.
        static const unsigned default_max_hotspots = 1024;
        
        /// \brief Optional\n
        /// Usage: --min-peak-amplitude -a [float >= 0]\n
        /// The number passed here must be less than \ref
        /// args_t::max_peak_amplitude "max_peak_amplitude".
        /// See \ref args_t::default_min_peak_amplitude "default value" and \ref
        /// power_map_state_t::min_peak_amplitude
        float min_peak_amplitude;
        
        /// See \ref power_map_state_t::min_peak_amplitude and arg_status_t
        arg_status_t min_peak_amplitude_status;
        
        /// \brief If the associated argument is not passed on the command line,
        /// this value is assumed.  Note that passing and \ref arg_status_t
        /// "INVALID" or mal-formed argument still generates an error and the
        /// default value will not be assumed.
        static constexpr const float default_min_peak_amplitude = 10.0f;
        
        /// \brief Optional\n
        /// Usage: --max-peak-amplitude -A [float >= 0]\n
        /// The number passed here must be greater than \ref
        /// args_t::min_peak_amplitude "min_peak_amplitude".
        /// See \ref args_t::default_max_peak_amplitude "default value" and \ref
        /// power_map_state_t::max_peak_amplitude
        float max_peak_amplitude;
        
        /// See \ref power_map_state_t::max_peak_amplitude and arg_status_t
        arg_status_t max_peak_amplitude_status;
        
        /// \brief If the associated argument is not passed on the command line,
        /// this value is assumed.  Note that passing and \ref arg_status_t
        /// "INVALID" or mal-formed argument still generates an error and the
        /// default value will not be assumed.
        static constexpr const float default_max_peak_amplitude = 50.0f;
        
        /// \brief Optional\n
        /// Usage: --min-stddev -s [float >= 0]\n
        /// The number passed here must be less than \ref args_t::max_stddev
        /// "max_stddev".  See \ref args_t::default_min_stddev "default value"
        /// and \ref power_map_state_t::min_stddev
        float min_stddev;
        
        /// See \ref power_map_state_t::min_stddev and arg_status_t
        arg_status_t min_stddev_status;
        
        /// \brief If the associated argument is not passed on the command line,
        /// this value is assumed.  Note that passing and \ref arg_status_t
        /// "INVALID" or mal-formed argument still generates an error and the
        /// default value will not be assumed.
        static constexpr const float default_min_stddev = 5.0f;
        
        /// \brief Optional\n
        /// Usage --max-stddev -S [float >= 0]\n
        /// The number passed here must be greater than \ref args_t::min_stddev
        /// "min_stddev".  See \ref args_t::default_max_stddev "default value"
        /// and \ref power_map_state_t::max_stddev
        float max_stddev;
        
        /// See \ref power_map_state_t::max_stddev and arg_status_t
        arg_status_t max_stddev_status;
        
        /// \brief If the associated argument is not passed on the command line,
        /// this value is assumed.  Note that passing and \ref arg_status_t
        /// "INVALID" or mal-formed argument still generates an error and the
        /// default value will not be assumed.
        static constexpr const float default_max_stddev = 15.0f;
        
        /// \brief Optional\n
        /// Usage: --min-aging-rate -r [float > 0]\n
        /// The number passed here must be less than \ref args_t::max_aging_rate
        /// "max_aging_rate".  See \ref args_t::default_min_aging_rate "default
        /// value" and \ref power_map_state_t::min_aging_rate
        float min_aging_rate;
        
        /// See \ref power_map_state_t::min_aging_rate and arg_status_t
        arg_status_t min_aging_rate_status;
        
        /// \brief If the associated argument is not passed on the command line,
        /// this value is assumed.  Note that passing and \ref arg_status_t
        /// "INVALID" or mal-formed argument still generates an error and the
        /// default value will not be assumed.
        static constexpr const float default_min_aging_rate = 5.0f;
        
        /// \brief Optional\n
        /// Usage: --max-aging-rate -R [float > 0]\n
        /// The number passed here must be greater than \ref
        /// args_t::min_aging_rate "min_aging_rate".  See \ref
        /// args_t::default_max_aging_rate "default value" and \ref
        /// power_map_state_t::max_aging_rate
        float max_aging_rate;
        
        /// See \ref power_map_state_t::max_aging_rate and arg_status_t
        arg_status_t max_aging_rate_status;
        
        /// \brief If the associated argument is not passed on the command line,
        /// this value is assumed.  Note that passing and \ref arg_status_t
        /// "INVALID" or mal-formed argument still generates an error and the
        /// default value will not be assumed.
        static constexpr const float default_max_aging_rate = 20.0f;
        
    protected:
        
        /// \brief Scan through the elements in argv looking for a particular argument
        /// \return The index into argv where either long_parameter or short_parameter
        /// is found, whichever came first.  Returns UINT_MAX if neither the long nor
        /// short version were found.
        static unsigned search_argv
        (
            
            /// [in] Number of elements in argv.  Same as from \ref main
            unsigned argc,
            
            /// [in] Array of tokenzied arguments passed to \ref main
            char const* const* argv,
            
            /// [in] Long version of a command line parameter (such as "--width").
            // TODO Please change std::experimental::string_view to std::string_view
            // when it's officially released.
            std::experimental::string_view long_parameter,
            
            /// [in] Short version of a command line parameter (such as "-w").
            // TODO Please change std::experimental::string_view to std::string_view
            // when it's officially released.
            std::experimental::string_view short_parameter
            
        );

        /// \brief Helper function for \ref args_t::parse "parse".  Searches
        /// for an argument and does a bunch of validation checks on the
        /// argument before returning it.
        /// \return The same value that is written back through the arg_status
        /// argument.
        static arg_status_t parse_unsigned_argument
        (
            
            /// [in] Number of elements in argv.  Same as from \ref main
            unsigned argc,
            
            /// [in] Array of tokenzied arguments passed to \ref main
            char const* const* argv,
            
            /// [in] Long version of a command line parameter (such as "--width").
            // TODO Please change std::experimental::string_view to std::string_view
            // when it's officially released.
            std::experimental::string_view long_parameter,
            
            /// [in] Short version of a command line parameter (such as "-w").
            // TODO Please change std::experimental::string_view to std::string_view
            // when it's officially released.
            std::experimental::string_view short_parameter,
            
            /// [out] See \ref arg_status_t.  This is always written back with the
            /// parsing status for the given function call.
            arg_status_t* arg_status,
            
            /// [out] If an argument is successfully parsed, the numeric value after
            /// that argument is written back.  This pointer is not written back if
            /// parsing failed.  For example, if argv[4] == "--width" and argv[5] == "42"
            /// then you could expect 42 to be written back via the parameter_value
            /// pointer.
            unsigned* parameter_value,
            
            /// See \ref args_t::parse "parse".
            bool* consumed
            
        );
        
        /// \brief Same as \ref args_t::parse_unsigned_argument
        /// "parse_unsigned_argument" except for floats.  This function is not
        /// templatized to avoid problems in which the compiler creates a
        /// redundant copy of this function for every file that includes this
        /// header.
        static arg_status_t parse_float_argument
        (
            
            /// See \ref args_t::parse_unsigned_argument
            unsigned argc,
            
            /// See \ref args_t::parse_unsigned_argument
            char const* const* argv,
            
            /// See \ref args_t::parse_unsigned_argument
            // TODO Please change std::experimental::string_view to std::string_view
            // when it's officially released.
            std::experimental::string_view long_parameter,
            
            /// See \ref args_t::parse_unsigned_argument
            // TODO Please change std::experimental::string_view to std::string_view
            // when it's officially released.
            std::experimental::string_view short_parameter,
            
            /// See \ref args_t::parse_unsigned_argument
            arg_status_t* arg_status,
            
            /// See \ref args_t::parse_unsigned_argument
            float* parameter_value,
            
            /// See \ref args_t::parse "parse".
            bool* consumed
            
        );
        
};

#endif // header guard

