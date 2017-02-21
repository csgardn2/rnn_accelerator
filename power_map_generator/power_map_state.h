/// \file
/// File Name:                      power_map_state.cpp \n
/// Date created:                   Mon Feb 20 2017 \n
/// Engineers:                      Conor Gardner \n
/// Special Compile Instructions:   --std=c++11 \n
/// Compiler:                       g++ \n
/// Target OS:                      Ubuntu Linux 16.04 \n
/// Target architecture:            x86 64-bit \n */
/// See html/index.html for documentation.
 
#ifndef HEADER_GUARD_POWER_MAP_STATE
#define HEADER_GUARD_POWER_MAP_STATE

#include <random>
#include <vector>
#include "hotspot.h"

/// \brief Used to track the locations and life stage of a bunch of hotspots
/// on a 2D power grid.  This class is designed to be used in conjunction with
/// \ref generate_power_map().
class power_map_state_t
{
    
    public:
        
        /// \brief Default constructor.
        power_map_state_t() = default;
        
        /// \brief Copy constructor
        power_map_state_t(const power_map_state_t& rhs) = default;
        
        /// \brief Initialization constructor.
        inline power_map_state_t
        (
            unsigned width_,
            unsigned height_,
            unsigned max_hotspots_,
            float min_peak_amplitude_,
            float max_peak_amplitude_,
            float min_stddev_,
            float max_stddev_,
            float min_aging_rate_,
            float max_aging_rate_
        )
          : width(width_),
            height(height_),
            max_hotspots(max_hotspots_),
            min_peak_amplitude(min_peak_amplitude_),
            max_peak_amplitude(max_peak_amplitude_),
            min_stddev(min_stddev_),
            max_stddev(max_stddev_),
            min_aging_rate(min_aging_rate_),
            max_aging_rate(max_aging_rate_)
        {
            // Intentionally left blank
        }
        
        /// \brief Generate some additional randomly generated hotspots and
        /// bring the total hotspot count closer to \ref
        /// power_map_state_t::max_hotspots "max_hotspots".
        /// \return The number of hotspots that were added.  It is possible
        /// for this value to be 0 if \ref power_map_state_t::hotspots
        /// "hotspots" was already at capacity.
        unsigned add_random_hotspots();
        
        /// \brief Generate a new random hotspot and overwrite whatever 
        /// currently exists at \ref power_map_state_t::hotspots "hotspots[ix]".
        void overwrite_random_hotspot(unsigned ix);
        
        /// \brief The number of columns in the 2D power map grid used in
        /// hotspot.cu from the rodinia benchmark suite.
        unsigned width;
        
        /// \brief The number of rows in the 2D power map grid used in
        /// hotspot.cu from the rodinia benchmark suite.
        unsigned height;
        
        /// \brief See \ref power_map_state_t "class description".
        /// Sets an upper limit to the number of hotspots which may be 
        /// appended to the power grid.
        unsigned max_hotspots;
        
        /// \brief Each time a new hotspot is created, a random value between
        /// this and \ref power_map_state_t::max_peak_amplitude
        /// "max_peak_amplitude" is used to intialize the new \ref
        /// hotspot_t::peak_amplitude "peak_amplitude".
        float min_peak_amplitude;
        
        /// \brief Each time a new hotspot is created, a random value between
        /// \ref power_map_state_t::min_peak_amplitude "mix_peak_amplitude" and
        /// this is used to intialize the new \ref hotspot_t::peak_amplitude
        /// "peak_amplitude".
        float max_peak_amplitude;
        
        /// \brief Each time a new hotspot is created, a random value between
        /// this and \ref power_map_state_t::max_stddev "max_stddev" is used to
        /// initialize \ref hotspot_t::stddev_x "stddev_x" and \ref
        /// hotspot_t::stddev_y "stddev_y" members of the new hotspot.
        float min_stddev;
        
        /// \brief Each time a new hotspot is created, a random value between
        /// \ref power_map_state_t::min_stddev "min_stddev" and this is used to
        /// initialize \ref hotspot_t::stddev_x "stddev_x" and \ref
        /// hotspot_t::stddev_y "stddev_y" members of the new hotspot.
        float max_stddev;
        
        /// \brief Each time a new hotspot is created, a random value between
        /// this and \ref power_map_state_t::max_aging_rate "max_againg_rate" is
        /// used to initialize the new \ref hotspot_t::aging_rate "aging_rate".
        float min_aging_rate;
        
        /// \brief Each time a new hotspot is created, a random value between
        /// \ref power_map_state_t::min_aging_rate "min_againg_rate" and this is
        /// used to initialize the new \ref hotspot_t::aging_rate "aging_rate".
        float max_aging_rate;
        
        /// \brief A random number generator for spawning new hotspots.  We must
        /// keep this as a class member because a default-constructed generator
        /// will always generate the same sequence, and we want to randomly
        /// generate different stats for each hotspot when they spawn in.
        std::default_random_engine generator;
        
        /// \brief Tracks the amplitudes, XY positions, standard deviations,
        /// life stage, peak amplitude, and aging rate of each hot patch on
        /// the power grid.
        std::vector<hotspot_t> hotspots;
    
};

#endif // header guard

