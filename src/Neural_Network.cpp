#include "Neural_Network.h"

double nn::derivative(double x, double dx, std::function<double(double)> func)
{
    // Compute d/dx[func(*first)] using a three-point
    // central difference rule of O(dx^6).

    const double dx1 = dx;
    const double dx2 = dx1 * 2;
    const double dx3 = dx1 * 3;

    const double m1 = (func(x + dx1) - func(x - dx1)) / 2;
    const double m2 = (func(x + dx2) - func(x - dx2)) / 4;
    const double m3 = (func(x + dx3) - func(x - dx3)) / 6;

    const double fifteen_m1 = 15 * m1;
    const double six_m2     =  6 * m2;
    const double ten_dx1    = 10 * dx1;

    return ((fifteen_m1 - six_m2) + m3) / ten_dx1;
}