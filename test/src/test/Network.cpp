#include "test/Network.h"

using nn::test::Network;

TEST_F(Network, Constructor_WillLayersHaveRightSize)
{
    ASSERT_EQ(initialNumberOfLayers, network.getCapacityOfLayers());
}

TEST_F(Network, pushLayer_WillThrowIfFirstLayerIsNotOfTypeLayerBegin)
{
    ASSERT_THROW(network.pushLayer(std::make_shared<nn::Layer<nn::Neuron>>(5)),
                 nn::Network::InvalidFirstLayerException);
    ASSERT_NO_THROW(network.pushLayer(std::make_shared<nn::BeginLayer<nn::BeginNeuron>>(5)));
}

TEST_F(Network, pushLayer_WillSizeIncrease)
{
    network.pushLayer(std::make_shared<nn::BeginLayer<nn::BeginNeuron>>(5));
    ASSERT_EQ(network.getNumberOfLayers(), 1);
    for (int i = 1; i < 5; i++)
    {
        network.pushLayer(std::make_shared<nn::Layer<nn::Neuron>>(i));
        ASSERT_EQ(network.getNumberOfLayers(), i + 1);
    }
}

TEST_F(Network, SetInput_WillFirstLayersValuesBeSetCorrectly)
{
    network.pushLayer(std::make_shared<nn::BeginLayer<nn::BeginNeuron>>(5));
    for (int i = 1; i <= 5; i++)
        network.pushLayer(std::make_shared<nn::Layer<nn::Neuron>>(5));

    std::vector<double> inputs = {0.1, 0.1, 0.1, 0.1, 0.1};
    network.setInputs(inputs);

    for (int i = 0; i < inputs.size(); i++)
        ASSERT_EQ(network.getLayer(0)->getNeurons()[i]->getValue(), inputs[i]);
}