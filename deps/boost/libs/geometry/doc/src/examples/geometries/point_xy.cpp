// Boost.Geometry (aka GGL, Generic Geometry Library)
// QuickBook Example

// Copyright (c) 2011-2012 Barend Gehrels, Amsterdam, the Netherlands.

// Use, modification and distribution is subject to the Boost Software License,
// Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)

//[point_xy
//` Declaration and use of the Boost.Geometry model::d2::point_xy, modelling the Point Concept

#include <iostream>
#include <boost/geometry.hpp>
#include <boost/geometry/geometries/point_xy.hpp>

namespace bg = boost::geometry;

int main()
{
    bg::model::d2::point_xy<double> point1;
    bg::model::d2::point_xy<double> point2(1.0, 2.0); /*< Construct, assigning coordinates. >*/

    bg::set<0>(point1, 1.0); /*< Set a coordinate, generic. >*/
    point1.y(2.0); /*< Set a coordinate, class-specific ([*Note]: prefer `bg::set()`). >*/

    double x = bg::get<0>(point1); /*< Get a coordinate, generic. >*/
    double y = point1.y(); /*< Get a coordinate, class-specific ([*Note]: prefer `bg::get()`). >*/

    std::cout << x << ", " << y << std::endl;
    return 0;
}

//]


//[point_xy_output
/*`
Output:
[pre
1, 2
]
*/
//]
