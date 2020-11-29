#ifndef NEURONAL_NETWORK_TEST_LAYER_H
#define NEURONAL_NETWORK_TEST_LAYER_H

#include "test/pch.h"

#include "Neural_Network.h"

namespace nn::test
{
    class Layer : public testing::Test
    {
    public:
        int numberOfNeurons = 4;
        nn::Layer<nn::Neuron> layer = {numberOfNeurons};
        nn::InputLayer<nn::InputNeuron> beginLayer{};

        std::vector<std::shared_ptr<nn::abs::InputNeuron>> neurons = {std::make_shared<nn::InputNeuron>(0.0),
                                                                      std::make_shared<nn::InputNeuron>(1.0),
                                                                      std::make_shared<nn::InputNeuron>(2.0),
                                                                      std::make_shared<nn::InputNeuron>(3.0),
                                                                      std::make_shared<nn::InputNeuron>(4.0),
                                                                      std::make_shared<nn::InputNeuron>(5.0),
                                                                      std::make_shared<nn::InputNeuron>(6.0),
                                                                      std::make_shared<nn::InputNeuron>(7.0),
                                                                      std::make_shared<nn::InputNeuron>(8.0),
                                                                      std::make_shared<nn::InputNeuron>(9.0)};

        static std::shared_ptr<nn::abs::Activation> getActivation(std::shared_ptr<nn::abs::Neuron> n);
    };
}

#endif //NEURONAL_NETWORK_TEST_LAYER_H
