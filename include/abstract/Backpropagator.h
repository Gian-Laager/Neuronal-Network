#ifndef NEURONAL_NETWORK_ABSTRACT_BACKPROPAGATOR_H
#define NEURONAL_NETWORK_ABSTRACT_BACKPROPAGATOR_H

#include "abstract/LossFunction.h"

namespace nn::abs
{
    class Network;

    class Backpropagator
    {
    public:
        virtual void fit(nn::abs::Network* n, const std::vector<std::vector<double>>& x,
                                              const std::vector<std::vector<double>>& y,
                                              std::shared_ptr<nn::abs::LossFunction> lossF,
                                              long batchSize, long epochs = 1) = 0;
    };
}

#endif //NEURONAL_NETWORK_ABSTRACT_BACKPROPAGATOR_H
