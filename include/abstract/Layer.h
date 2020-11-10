#ifndef NEURONAL_NETWORK_ABSTRACT_LAYER_H
#define NEURONAL_NETWORK_ABSTRACT_LAYER_H

#include "abstract/Neuron.h"

namespace nn::abs
{
    class Layer
    {
    public:
        virtual int getSize() const = 0;

        virtual void connect(Layer* l) = 0;

        virtual std::vector<std::shared_ptr<nn::abs::Neuron>> getNeurons() = 0;
    };
}

#endif //NEURONAL_NETWORK_ABSTRACT_LAYER_H
