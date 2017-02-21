/// \file
/// File Name:                      hotspot.cpp \n
/// Date created:                   Fri Feb 17 2017 \n
/// Engineers:                      Conor Gardner \n
/// Special Compile Instructions:   --std=c++11 \n
/// Compiler:                       g++ \n
/// Target OS:                      Ubuntu Linux 16.04 \n
/// Target architecture:            x86 64-bit \n */
/// See html/index.html for documentation.
 
#ifndef HEADER_GUARD_HOTSPOT
#define HEADER_GUARD_HOTSPOT

/// \brief Tracks wheather a hotspot's amplitude is increasing or decreasing.
/// See \ref power_map_state_t.
enum class hotspot_stage_t
{
    GROWING,
    SHRINKING,
    DEAD
};

/// \brief Class for storing the parameters for a gaussian shaped hotspot
/// to be drawn onto a power map.
///
/// Each hotspot is a gaussian distribution with a life cycle in which it spawns
/// onto the power map at a random location with zero initial amplitude, \ref
/// hotspot_stage_t "grows" by \ref hotspot_t::aging_rate "aging rate" until
/// the hotspot's \ref hotspot_t::amplitude "amplitude" reaches or exceeds
/// its \ref hotspot_t::peak_amplitude "peak amplitude", then \ref
/// hotspot_stage_t "shrinks" at the same rate until it dissapears and is
/// replaced by a new hotspot.
class hotspot_t
{
        
    public:
        
        /// \brief Default constructor
        hotspot_t() = default;
        
        /// \brief Copy constructor
        hotspot_t(const hotspot_t& rhs) = default;
        
        /// \brief Initialization constructor
        inline hotspot_t
        (
            float amplitude_,
            float mean_x_,
            float mean_y_,
            float stddev_x_,
            float stddev_y_,
            float peak_amplitude_,
            float aging_rate_
        )
          : amplitude(amplitude_),
            mean_x(mean_x_),
            mean_y(mean_y_),
            stddev_x(stddev_x_),
            stddev_y(stddev_y_),
            stage(hotspot_stage_t::GROWING),
            peak_amplitude(peak_amplitude_),
            aging_rate(aging_rate_)
        {
            // Intentionally left blank
        }
        
        /// \brief Move the hotspot one step forward along its normal lifecycle.
        /// \return The new value of \ref hotspot_t::stage "stage" at the end of
        /// the function call.
        /// 
        /// Detailed function behavior:
        ///     1. If this hotspot was \ref hotspot_stage_t "GROWING" before
        ///     this function call, then \ref hotspot_t::aging_rate "aging_rate"
        ///     is added to \ref hotspot_t::amplitude "amplitude".  If the new
        ///     amplitude >= \ref hotspot_t::peak_amplitude "peak_amplitude",
        ///     then \ref hotspot_t::stage "stage" is overwritten with \ref
        ///     hotspot_stage_t "SHRINKING".
        ///     2. If this hotspot was \ref hotspot_stage_t "SHRINKING" before
        ///     this function call, then \ref hotspot_t::aging_rate "aging_rate"
        ///     is subtracted from \ref hotspot_t::peak_amplitude "amplitude".
        ///     Of the new amplitude <= 0.0f, then \ref hotspot_t::stage "stage"
        ///     is overwritten with \ref hotspot_stage_t "DEAD" and amplitude is
        ///     clamped to 0.0f.
        ///     3. If this hotspot was \ref hotspot_stage_t "DEAD" before this
        ///     function call, then this function returns immediatly without
        ///     performing any operation.
        hotspot_stage_t advance_age();
        
        /// \brief Evaluate the gaussian function of this hotspot at a 2D
        /// certain point.
        float operator()(float x, float y) const;
        
        /// Calculate the value X such that threshold == \ref
        /// hotspot_t::operator()(float x, float y) const "operator()(X, 0)"
        float calculate_falloff_x(float threshold) const;
        
        /// Calculate the value Y such that threshold == \ref
        /// hotspot_t::operator()(float x, float y) const "operator()(0, Y)"
        float calculate_falloff_y(float threshold) const;
        
        /// \brief Assignment operator
        hotspot_t& operator=(const hotspot_t& rhs) = default;
        
        /// \brief Used to scale up the height of the gaussian at all points.
        /// For a normalized gaussian, this will be 1.0f.
        float amplitude;
        
        /// \brief Mean value (usually labeled with the Greek letter mu) in the
        /// X direction.  Intuitivly, that value is the X-coordinate of the
        /// peak of the hotspot.  Increasing or decreasing this value slides
        /// the bell curve right or left respectivly.
        float mean_x;
        
        /// \brief Mean value (usually labeled with the Greek letter mu) in the
        /// Y direction.  Intuitivly, that value is the Y-coordinate of the
        /// peak of the hotspot.  Increasing or decreasing this value slides
        /// the bell curve up or down respectivly.
        float mean_y;
        
        /// \brief Standard deviation (usually labeled with the Greek letter
        /// sigma) along the X axis.  Intuitivly, this value is the width of the
        /// curve along the X direction.  Increasing or decreasing this value
        /// makes the curve fatter or thinner respectivly.
        float stddev_x;
        
        /// \brief Standard deviation (usually labeled with the Greek letter
        /// sigma) along the Y axis.  Intuitivly, this value is the height of
        /// the curve along the Y direction.  Increasing or decreasing this 
        /// value makes the curve fatter or thinner respectivly.
        float stddev_y;
        
        /// \brief Tracks wheather the the current hotspot is growing or
        /// shrinking.
        hotspot_stage_t stage;
        
        /// \brief See \ref hotspot_t "class description".
        float peak_amplitude;
        
        /// \brief This is the amount that is added or subtracted from 
        /// \ref hotspot_t::amplitude "amplitude" during each iteration (call to 
        /// \ref generate_power_map)
        float aging_rate;
        
};

#endif // header guard

