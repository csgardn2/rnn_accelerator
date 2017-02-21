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

#include <iostream>
#include <string>

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
enum class arg_error_code_t : unsigned char
{
    
    SUCCESS,
    WIDTH_NOT_PASSED,
    WIDTH_WITHOUT_ARGUMENT,
    WIDTH_INVALID,
    HEIGHT_NOT_PASSED,
    HEIGHT_WITHOUT_ARGUMENT,
    HEIGHT_INVALID,
    TIME_STEPS_NOT_PASSED,
    TIME_STEPS_WITHOUT_ARGUMENT,
    TIME_STEPS_INVALID,
    BASE_FILENAME_NOT_PASSED,
    BASE_FILENAME_WITHOUT_ARGUEMENT,
    
    /// \brief This element MUST be the last enum.
    NUM_ERROR_CODES
    
};

/// \brief Class which contains both an \ref arg_error_code_t "error code" and a
/// human readable message describing that error code.  This class is designed
/// to be the return value of \ref args_t::parse "parse"
class arg_error_t
{
    
    public:
        
        /// \brief Default constructor
        inline arg_error_t() = default;
        
        /// \brief Copy constructor
        inline arg_error_t(const arg_error_t& rhs) = default;
        
        /// \brief Initialization constructor.
        inline arg_error_t(arg_error_code_t error_code_)
        {
            if (unsigned(error_code_) < unsigned(arg_error_code_t::NUM_ERROR_CODES))
            {
                this->error_code = error_code_;
                this->error_string = arg_error_t::enum_to_string(error_code_);
            }
        }
        
        /// \brief Returns true if the two \ref error_code "error codes" of the two
        /// objects match, ignoring the \ref error_string "error strings".
        inline bool operator==(const arg_error_t& rhs)
        {
            return this->error_code == rhs.error_code;
        }
        
        /// \brief Returns true if the error code of this object matches the given
        /// error code.
        inline bool operator==(arg_error_code_t rhs)
        {
            return this->error_code == rhs;
        }
        
        /// \brief See \ref arg_error_t::operator==(const arg_error_t&)
        inline bool operator!=(const arg_error_t& rhs)
        {
            return !(*this == rhs);
        }
        
        /// \brief See \ref arg_error_t::operator==(arg_error_code_t)
        inline bool operator!=(arg_error_code_t rhs)
        {
            return !(*this == rhs);
        }
        
        /// \brief Return a human readable string corresponding to an error code.
        static const char* enum_to_string(arg_error_code_t error_code_);
        
        /// \brief See arg_error_code_t.
        arg_error_code_t error_code;
        
        /// \brief A human readable message explaining why this error occurred.
        const char* error_string;
        
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
            base_filename_status(arg_status_t::NOT_FOUND)
        {
            // Intentionally left blank
        }
        
        /// \brief See \ref parse
        inline args_t(unsigned argc, char** argv, bool* consumed = nullptr)
        {
            this->parse(argc, argv, consumed);
        }
        
        /// \brief Parse the command line parameters passed to \ref main and
        /// initialize all relevant args_t variables if the corresponding
        /// parameter is passed.
        /// \return \ref arg_error_code_t "SUCCESS" if sufficient parameters
        /// were successfully parsed or defaulted to execute the rest of the
        /// program.
        arg_error_t parse
        (
            
            /// [in] The same as argc from \ref main.  The number of elements
            /// in the argv array and the optional consumed array.
            unsigned argc,
            /// [in] Array of strings passed from \ref main.  Each is a token
            /// passed as a command line parameter.
            char** argv,
            
            /// [out].  Optional.  If you want to verify which command line
            /// parameters were recognized and consumed, allocate an pass an
            /// array of bools the same size as argv here.  If a command line
            /// parameter (and possibly its argument) were consumed, their
            /// corresponding element in the consumed array is set to true.
            /// Note that elements are set to true even if an argument was
            /// invalid.
            bool* consumed = nullptr
            
        );
        
        /// \brief Required.\n
        /// Usage: --width -w [integer >= 0]\n
        /// The number of columns in the output power map.
        unsigned width;
        
        /// \brief Required.\n
        /// Usage: --height -h [integer >= 0]\n
        /// The number of rows in the output power map.
        unsigned height;
        
        /// \brief Required.\n
        /// Usage: --time-steps -t [integer >= 0]\n
        /// The number output power maps (files) to generate.
        unsigned time_steps;
        
        /// \brief Required.\n
        /// Usage: --base-filename -o [string]\n
        /// The base part of a filename which will be used to save power
        /// maps.
        /// Example:
        /// The user passes '--base-filename lard' and '--time-steps 4' on the
        /// command line.  The output files will be named 'lard_0.txt',
        /// 'lard_1.txt', 'lard_2.txt', and 'lard_3.txt'.
        std::string base_filename;
        
        /// \brief see \ref width
        arg_status_t width_status;
        
        /// \brief see \ref height
        arg_status_t height_status;
        
        /// \brief see \ref time_steps
        arg_status_t time_steps_status;
        
        /// \brief see \ref base_filename
        arg_status_t base_filename_status;
        
};

#endif // header guard

