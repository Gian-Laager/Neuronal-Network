#ifndef NEURONAL_NETWORK_TEST_NEURON_H
#define NEURONAL_NETWORK_TEST_NEURON_H

#include "test/pch.h"

#include <Neuron.h>

namespace nn::test
{
    class Neuron : public testing::Test
    {
    public:
        std::vector<nn::Connection*> connectionsNextLayer = {new Connection{nullptr}, new Connection{nullptr}, new Connection{nullptr}, new Connection{nullptr}, new Connection{nullptr}};
        std::vector<nn::Connection*> connectionsPreviousLayer = {new Connection{nullptr}, new Connection{nullptr}, new Connection{nullptr}, new Connection{nullptr}};
        nn::Neuron n;

        void SetUp() override;
    };
}
#endif //NEURONAL_NETWORK_NEURON_H
