#include "test/Network.h"

using nn::test::Network;

TEST_F(Network, Constructor_WillLayersHaveRightSize)
{
    ASSERT_EQ(initialNumberOfLayers, network.getCapacityOfLayers());
    ASSERT_TRUE(getBackpropagator(&network).get());
}

TEST_F(Network, pushLayer_WillThrowIfFirstLayerIsNotOfTypeLayerBegin)
{
    ASSERT_THROW(network.pushLayer(std::make_shared<nn::Layer<nn::Neuron>>(5)),
                 nn::Network::InvalidFirstLayerException);
    ASSERT_NO_THROW(network.pushLayer(std::make_shared<nn::InputLayer<nn::InputNeuron>>(5)));
}

TEST_F(Network, pushLayer_WillSizeIncrease)
{
    network.pushLayer(std::make_shared<nn::InputLayer<nn::InputNeuron>>(5));
    ASSERT_EQ(network.getNumberOfLayers(), 1);
    for (int i = 1; i < 5; i++)
    {
        network.pushLayer(std::make_shared<nn::Layer<nn::Neuron>>(i));
        ASSERT_EQ(network.getNumberOfLayers(), i + 1);
    }
}

TEST_F(Network, SetInput_WillFirstLayersValuesBeSetCorrectly)
{
    network.pushLayer(std::make_shared<nn::InputLayer<nn::InputNeuron>>(5));
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
    network.pushLayer(std::make_shared<nn::InputLayer<nn::InputNeuron>>(firstLayerSize));
    std::vector<double> biasesInvalid(firstLayerSize + 1);
    ASSERT_THROW(network.setBias(0, biasesInvalid),
                 nn::InputLayer<nn::InputNeuron>::IncompatibleVectorException);
    std::vector<double> biasesValid(firstLayerSize);
    ASSERT_NO_THROW(network.setBias(0, biasesValid));
}

TEST_F(Network, SetBias_WillBiasBeSetCorrectly)
{
    int firstLayerSize = 5;
    int secondLayerSize = 3;
    network.pushLayer(std::make_shared<nn::InputLayer<nn::InputNeuron>>(firstLayerSize));
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
    class FirstLayerActivation : public nn::abs::Activation
    {
    public:
        double operator()(double z) override { return z - 1; }

        double derivative(double z) override { return 1; }
    };
    class SecondLayerActivation : public nn::abs::Activation
    {
    public:
        double operator()(double z) override { return 1 / z; }

        double derivative(double z) override { return 1; }
    };
    class ThirdLayerActivation : public nn::abs::Activation
    {
    public:
        double operator()(double z) override { return z * z; }

        double derivative(double z) override { return 1; }
    };

    nn::Network net{3};
    net.pushLayer(std::make_shared<nn::InputLayer<nn::InputNeuron>>(2, std::make_shared<FirstLayerActivation>()));
    net.pushLayer(std::make_shared<nn::Layer<nn::Neuron>>(3, std::make_shared<SecondLayerActivation>()));
    net.pushLayer(std::make_shared<nn::Layer<nn::Neuron>>(2, std::make_shared<ThirdLayerActivation>()));

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
#ifdef MessureTimeNeededToCalculate
    auto tp1 = std::chrono::high_resolution_clock::now();
#endif
    std::vector<double> resultCalc = net.calculate();
#ifdef MessureTimeNeededToCalculate
    auto tp2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsedTime = tp2 - tp1;

    std::cout << "Time spend on net.calculate: " << elapsedTime.count() * 1000 << " ms" << std::endl;
#endif

    std::vector<double> resultRight{130321.0 / 3025.0, 48.450036730945811};
    ASSERT_EQ(resultCalc.size(), net.getLayer(net.getSize() - 1)->getSize());
    ASSERT_EQ(resultCalc.size(), resultRight.size());

    for (int i = 0; i < resultRight.size(); i++)
        ASSERT_EQ(resultCalc[i], resultRight[i]);
}