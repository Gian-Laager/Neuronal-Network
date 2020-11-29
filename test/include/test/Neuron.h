#ifndef NEURONAL_NETWORK_TEST_NEURON_H
#define NEURONAL_NETWORK_TEST_NEURON_H

#include "test/pch.h"

#include <Neuron.h>

namespace nn::test
{
    class Neuron : public testing::Test
    {
    public:
        nn::Neuron n;
        std::map<std::shared_ptr<nn::abs::Neuron>, std::shared_ptr<nn::abs::Connection>> connectionsNextLayer = {
                {nullptr, std::make_shared<Connection>(nullptr)}, {nullptr, std::make_shared<Connection>(nullptr)}, 
                {nullptr, std::make_shared<Connection>(nullptr)}, {nullptr, std::make_shared<Connection>(nullptr)}, 
                {nullptr, std::make_shared<Connection>(nullptr)}};
        std::map<std::shared_ptr<nn::abs::Neuron>, std::shared_ptr<nn::abs::Connection>> connectionsPreviousLayer = {
                {nullptr, std::make_shared<Connection>(nullptr)}, {nullptr, std::make_shared<Connection>(nullptr)},
                {nullptr, std::make_shared<Connection>(nullptr)}, {nullptr, std::make_shared<Connection>(nullptr)},
                {nullptr, std::make_shared<Connection>(nullptr)}};

        void SetUp() override;
    };
}
#endif //NEURONAL_NETWORK_NEURON_H
