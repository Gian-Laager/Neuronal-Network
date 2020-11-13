#ifndef NEURONAL_NETWORK_Abstract_LOSS_FUNCTION_H
#define NEURONAL_NETWORK_Abstract_LOSS_FUNCTION_H

#include "pch.h"

namespace nn::abs
{
    class LossFunction
    {
    public:
        virtual double operator()(double yPred, double yTrue) = 0;

        virtual double derivative(double yPred, double yTrue) = 0;
    };
}

#endif //NEURONAL_NETWORK_Abstract_LOSS_FUNCTION_H
