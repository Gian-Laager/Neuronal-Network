#include "test/Backpropagator.h"

using namespace nn::test;

void nn::test::Backpropagator::SetUp()
{
    simpleNetwork = nn::Network{3};
    simpleNetwork.pushLayer(std::make_shared<nn::InputLayer<nn::InputNeuron>>(1));
    simpleNetwork.pushLayer(std::make_shared<nn::Layer<nn::Neuron>>(1));
    simpleNetwork.pushLayer(std::make_shared<nn::Layer<nn::Neuron>>(1));
    complexNetwork = nn::Network{3};
    complexNetwork.pushLayer(std::make_shared<nn::InputLayer<nn::InputNeuron>>(2));
    complexNetwork.pushLayer(std::make_shared<nn::Layer<nn::Neuron>>(3));
    complexNetwork.pushLayer(std::make_shared<nn::Layer<nn::Neuron>>(2));
}

TEST_F(Backpropagator, Fit_WillThrowExceptionWhenVectorIsInvalid)
{
    int simpleSizeInput = simpleNetwork.getInputLayerSize();
    int simpleSizeOut = simpleNetwork.getOutputLayerSize();

    std::vector<std::vector<double>> simpleValidInput = {std::vector<double>(simpleSizeInput)};
    std::vector<std::vector<double>> simpleInvalidInput = {std::vector<double>(simpleSizeInput + 3)};
    std::vector<std::vector<double>> simpleValidOut = {std::vector<double>(simpleSizeOut)};
    std::vector<std::vector<double>> simpleInvalidOut = {std::vector<double>(simpleSizeOut + 3)};

    ASSERT_THROW(simpleNetwork.fit(simpleInvalidInput, simpleValidOut,
                                   std::make_shared<nn::losses::MSE>(), 1), nn::Backpropagator::InvalidVectorSize);
    ASSERT_THROW(simpleNetwork.fit(simpleValidInput, simpleInvalidOut,
                                   std::make_shared<nn::losses::MSE>(), 1), nn::Backpropagator::InvalidVectorSize);
    ASSERT_THROW(simpleNetwork.fit(simpleInvalidInput, simpleInvalidOut,
                                   std::make_shared<nn::losses::MSE>(), 1), nn::Backpropagator::InvalidVectorSize);
    ASSERT_NO_THROW(simpleNetwork.fit(simpleValidInput, simpleValidOut,
                                      std::make_shared<nn::losses::MSE>(), 1));

    int complexSizeInput = complexNetwork.getInputLayerSize();
    int complexSizeOut = complexNetwork.getOutputLayerSize();

    std::vector<std::vector<double>> complexValidInput = {std::vector<double>(complexSizeInput)};
    std::vector<std::vector<double>> complexInvalidInput = {std::vector<double>(complexSizeInput + 1)};
    std::vector<std::vector<double>> complexValidOut = {std::vector<double>(complexSizeOut)};
    std::vector<std::vector<double>> complexInvalidOut = {std::vector<double>(complexSizeOut + 1)};

    ASSERT_THROW(complexNetwork.fit(complexInvalidInput, complexValidOut,
                                   std::make_shared<nn::losses::MSE>(), 1), nn::Backpropagator::InvalidVectorSize);
    ASSERT_THROW(complexNetwork.fit(complexValidInput, complexInvalidOut,
                                   std::make_shared<nn::losses::MSE>(), 1), nn::Backpropagator::InvalidVectorSize);
    ASSERT_THROW(complexNetwork.fit(complexInvalidInput, complexInvalidOut,
                                   std::make_shared<nn::losses::MSE>(), 1), nn::Backpropagator::InvalidVectorSize);
    ASSERT_NO_THROW(complexNetwork.fit(complexValidInput, complexValidOut,
                                      std::make_shared<nn::losses::MSE>(), 1));
}