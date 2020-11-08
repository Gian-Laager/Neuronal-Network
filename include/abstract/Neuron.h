#ifndef NEURONAL_NETWORK_ABSTRACT_NEURON_H
#define NEURONAL_NETWORK_ABSTRACT_NEURON_H

#include "pch.h"

namespace nn::abs
{
    struct Connection
    {
        double w = 0.0;
        double b = 0.0;
    };

    class Neuron
    {
    public:
        virtual std::vector<std::shared_ptr<nn::abs::Connection>> getConnectionsNextLayer() = 0;

        virtual std::vector<std::shared_ptr<nn::abs::Connection>> getConnectionsPreviousLayer() = 0;

        virtual void connect(Neuron* n) = 0;
    };
}

#endif //NEURONAL_NETWORK_ABSTRACT_NEURON_H
