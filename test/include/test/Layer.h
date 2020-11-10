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
        nn::BeginLayer<nn::BeginNeuron> beginLayer{};

        std::vector<std::shared_ptr<nn::abs::BeginNeuron>> neurons = {std::make_shared<nn::BeginNeuron>(0.0),
                                                                      std::make_shared<nn::BeginNeuron>(1.0),
                                                                      std::make_shared<nn::BeginNeuron>(2.0),
                                                                      std::make_shared<nn::BeginNeuron>(3.0),
                                                                      std::make_shared<nn::BeginNeuron>(4.0),
                                                                      std::make_shared<nn::BeginNeuron>(5.0),
                                                                      std::make_shared<nn::BeginNeuron>(6.0),
                                                                      std::make_shared<nn::BeginNeuron>(7.0),
                                                                      std::make_shared<nn::BeginNeuron>(8.0),
                                                                      std::make_shared<nn::BeginNeuron>(9.0)};

        static std::function<double(double)> getActivation(nn::Neuron* n);
    };
}

#endif //NEURONAL_NETWORK_TEST_LAYER_H
