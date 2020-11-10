#ifndef NEURONAL_NETWORK_TEST_LAYER_H
#define NEURONAL_NETWORK_TEST_LAYER_H

#include "test/pch.h"

#include "Layer.h"
#include "Neuron.h"

namespace nn::test
{
    class Layer : public testing::Test
    {
    public:
        int numberOfNeurons = 4;
        nn::Layer<nn::Neuron> layer = {numberOfNeurons};

        static std::function<double(double)> getActivation(nn::Neuron* n);
    };
}

#endif //NEURONAL_NETWORK_TEST_LAYER_H
