#ifndef NEURONAL_NETWORK_ABSTRACT_BACKPROPAGATOR_H
#define NEURONAL_NETWORK_ABSTRACT_BACKPROPAGATOR_H

#include "abstract/LossFunction.h"

namespace nn::abs
{
    class Network;

    class Backpropagator
    {
    public:
        virtual void initialize(nn::abs::Network* n, std::shared_ptr<nn::abs::LossFunction> lossF) = 0;

        virtual void fit(const std::vector<std::vector<double>>& x,
                         const std::vector<std::vector<double>>& y, int epochs, int batchSize) = 0;
        virtual void fit(const std::vector<std::vector<double>>& x,
                         const std::vector<std::vector<double>>& y, int epochs) = 0;
    };
}

#endif //NEURONAL_NETWORK_ABSTRACT_BACKPROPAGATOR_H
