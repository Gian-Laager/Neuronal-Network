#ifndef NEURONAL_NETWORK_LOSS_FUNCTIONS_H
#define NEURONAL_NETWORK_LOSS_FUNCTIONS_H

#include "abstract/LossFunction.h"

namespace nn::losses
{
    class MSE : public nn::abs::LossFunction
    {
        double operator()(double yPred, double yTrue) override
        {
            return pow(yPred - yTrue, 2);
        }

        double derivative(double yPred, double yTrue) override
        {
            return 2 * (yTrue - yPred);
        }
    };
}

#endif //NEURONAL_NETWORK_LOSS_FUNCTIONS_H
