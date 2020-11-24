#ifndef NEURONAL_NETWORK_TEST_BACKPROPAGATOR_H
#define NEURONAL_NETWORK_TEST_BACKPROPAGATOR_H

#include <Network.h>
#include <Neuron.h>
#include <Layer.h>
#include <LossFunctions.h>

namespace nn::test
{
    class Backpropagator : public testing::Test
    {
    public:
        nn::Network complexNetwork;
        nn::Network simpleNetwork;

        void SetUp() override;
    };
}

#endif //NEURONAL_NETWORK_TEST_BACKPROPAGATOR_H
