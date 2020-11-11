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

TEST_F(Network, SetBias_WillThrowExceptionWhenIncompatibleSizeOfVector)
{
    int firstLayerSize = 5;
    network.pushLayer(std::make_shared<nn::BeginLayer<nn::BeginNeuron>>(firstLayerSize));
    std::vector<double> biasesInvalid(firstLayerSize + 1);
    ASSERT_THROW(network.getLayer(0)->setBias(biasesInvalid),
                 nn::BeginLayer<nn::BeginNeuron>::IncompatibleVectorException);
    std::vector<double> biasesValid(firstLayerSize);
    ASSERT_NO_THROW(network.getLayer(0)->setBias(biasesValid));
}

TEST_F(Network, SetBias_WillBiasBeSetCorrectly)
{
    int firstLayerSize = 5;
    int secondLayerSize = 3;
    network.pushLayer(std::make_shared<nn::BeginLayer<nn::BeginNeuron>>(firstLayerSize));
    network.pushLayer(std::make_shared<nn::Layer<nn::Neuron>>(secondLayerSize));

    std::vector<double> firstBiases = {0.5, 0.2, -1, 2, -0.1};
    std::vector<double> secondBiases = {0.5, 0.2, -0.1};

    network.setBias(0, firstBiases);
    network.setBias(1, secondBiases);

    for (int i = 0; i < network.getLayer(0)->getSize(); i++)
        ASSERT_EQ(network.getLayer(0)->getNeurons()[i]->getB(), firstBiases[i]);

    for (int i = 0; i < network.getLayer(1)->getSize(); i++)
        ASSERT_EQ(network.getLayer(1)->getNeurons()[i]->getB(), secondBiases[i]);
}

TEST_F(Network, Callculate_WillTheValueBeCallculatedCorrectly)
{
    nn::Network net{3};
    net.pushLayer(std::make_shared<nn::BeginLayer<nn::BeginNeuron>>(2, [](double z) -> double { return z - 1; }));
    net.pushLayer(std::make_shared<nn::Layer<nn::Neuron>>(3, [](double z) -> double { return 1 / z; }));
    net.pushLayer(std::make_shared<nn::Layer<nn::Neuron>>(2, [](double z) -> double { return z * z; }));

    net.setBias(0, std::vector<double>{0.2, -3});
    net.setWeights(0, std::vector<std::map<nn::abs::Neuron*, double>>{
            std::map<nn::abs::Neuron*, double>{{net.getLayer(1)->getNeurons()[0].get(), 0.75},
                                               {net.getLayer(1)->getNeurons()[1].get(), 0.5},
                                               {net.getLayer(1)->getNeurons()[2].get(), 0.25}},

            std::map<nn::abs::Neuron*, double>{{net.getLayer(1)->getNeurons()[0].get(), 0.3},
                                               {net.getLayer(1)->getNeurons()[1].get(), 0.2},
                                               {net.getLayer(1)->getNeurons()[2].get(), 0.1}}});

    net.setBias(1, std::vector<double>{0.8, 0.3, 0.2});
    net.setWeights(1, std::vector<std::map<nn::abs::Neuron*, double>>{
            std::map<nn::abs::Neuron*, double>{{net.getLayer(2)->getNeurons()[0].get(), 0.2},
                                               {net.getLayer(2)->getNeurons()[1].get(), 0.4}},

            std::map<nn::abs::Neuron*, double>{{net.getLayer(2)->getNeurons()[0].get(), 0.6},
                                               {net.getLayer(2)->getNeurons()[1].get(), 0.8}},

            std::map<nn::abs::Neuron*, double>{{net.getLayer(2)->getNeurons()[0].get(), 1},
                                               {net.getLayer(2)->getNeurons()[1].get(), 1.2}}});

    net.setBias(2, std::vector<double>{-1.2, -0.1});

    net.setInputs(std::vector{0.5, 0.25});

    std::vector<double> resultCalc = net.calculate();
    std::vector<double> resultRight{130321.0 / 3025.0, 48.450036730945811};
    ASSERT_EQ(resultCalc.size(), net.getLayer(net.getSize() - 1)->getSize());
    ASSERT_EQ(resultCalc.size(), resultRight.size());

    for (int i = 0; i < resultRight.size(); i++)
        ASSERT_EQ(resultCalc[i], resultRight[i]);
}