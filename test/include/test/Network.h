#ifndef NEURONAL_NETWORK_TEST_NETWORK_H
#define NEURONAL_NETWORK_TEST_NETWORK_H


#include "pch.h"
#include "Neural_Network.h"

namespace nn::test
{
    class Network : public testing::Test
    {
    public:
        int initialNumberOfLayers = 5;
        nn::Network network{initialNumberOfLayers};

        std::shared_ptr<nn::abs::Backpropagator> getBackpropagator(nn::Network* n);
    };
}

std::shared_ptr<nn::abs::Backpropagator> nn::test::Network::getBackpropagator(nn::Network* n)
{
    return *(std::shared_ptr<nn::abs::Backpropagator>*) ((long) n + 40);
}

#endif //NEURONAL_NETWORK_NETWORK_H
