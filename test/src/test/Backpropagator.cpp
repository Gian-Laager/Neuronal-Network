#include "test/Backpropagator.h"

using namespace nn::test;

void nn::test::Backpropagator::SetUp()
{
    simpleNetwork = nn::Network{3};
    simpleNetwork.pushLayer(std::make_shared<nn::InputLayer<nn::InputNeuron>>(1));
    simpleNetwork.pushLayer(std::make_shared<nn::Layer<nn::Neuron>>(1));
    simpleNetwork.pushLayer(std::make_shared<nn::Layer<nn::Neuron>>(1));

    simpleNetwork.setWeights(0, std::vector<std::map<std::shared_ptr<nn::abs::Neuron>, double>>{std::map<nn::abs::Neuron*, double>{
            std::pair<std::shared_ptr<nn::abs::Neuron>, double>{simpleNetwork.getLayer(1)->getNeuron(0).get(), 2.0}}});

    simpleNetwork.setBias(0, {5.0});

    simpleNetwork.setWeights(1, std::vector<std::map<std::shared_ptr<nn::abs::Neuron>, double>>{std::map<std::shared_ptr<nn::abs::Neuron>, double>{
            std::pair<std::shared_ptr<nn::abs::Neuron>, double>{simpleNetwork.getLayer(2)->getNeuron(0).get(), 2.5}}});

    simpleNetwork.setBias(1, {5.0});
    simpleNetwork.setBias(2, {5.0});

    complexNetwork = nn::Network{3};
    complexNetwork.pushLayer(
            std::make_shared<nn::InputLayer<nn::InputNeuron>>(2, std::make_shared<nn::activations::Sigmoid>()));
    complexNetwork.pushLayer(std::make_shared<nn::Layer<nn::Neuron>>(3, std::make_shared<nn::activations::Sigmoid>()));
    complexNetwork.pushLayer(std::make_shared<nn::Layer<nn::Neuron>>(2, std::make_shared<nn::activations::Tanh>()));
}

TEST_F(Backpropagator, Fit_WillThrowExceptionWhenVectorIsInvalid)
{
    int simpleSizeInput = simpleNetwork.getInputLayerSize();
    int simpleSizeOut = simpleNetwork.getOutputLayerSize();

    std::vector<std::vector<double>> simpleValidInput = {std::vector<double>(simpleSizeInput)};
    std::vector<std::vector<double>> simpleInvalidInput = {std::vector<double>(simpleSizeInput + 3)};
    std::vector<std::vector<double>> simpleValidOut = {std::vector<double>(simpleSizeOut)};
    std::vector<std::vector<double>> simpleInvalidOut = {std::vector<double>(simpleSizeOut + 3)};
    std::vector<std::vector<double>> simpleOutUnequalSize = {std::vector<double>(simpleSizeOut),
                                                             std::vector<double>(simpleSizeOut)};
    std::vector<std::vector<double>> simpleInputUnequalSize = {std::vector<double>(simpleSizeInput)};
    simpleNetwork.initializeFitting(std::make_shared<nn::losses::MSE>());

    ASSERT_THROW(simpleNetwork.fit(simpleInvalidInput, simpleValidOut, 1.0, 1),
                 nn::Backpropagator::InvalidVectorSize);

    ASSERT_THROW(simpleNetwork.fit(simpleValidInput, simpleInvalidOut, 1.0, 1),
                 nn::Backpropagator::InvalidVectorSize);

    ASSERT_THROW(simpleNetwork.fit(simpleInvalidInput, simpleInvalidOut, 1.0, 1),
                 nn::Backpropagator::InvalidVectorSize);

    ASSERT_THROW(simpleNetwork.fit(simpleInputUnequalSize, simpleOutUnequalSize, 1.0, 1),
                 nn::Backpropagator::InvalidVectorSize);

    ASSERT_NO_THROW(simpleNetwork.fit(simpleValidInput, simpleValidOut, 1.0, 1));

    int complexSizeInput = complexNetwork.getInputLayerSize();
    int complexSizeOut = complexNetwork.getOutputLayerSize();

    std::vector<std::vector<double>> complexValidInput = {std::vector<double>(complexSizeInput)};
    std::vector<std::vector<double>> complexInvalidInput = {std::vector<double>(complexSizeInput + 1)};
    std::vector<std::vector<double>> complexValidOut = {std::vector<double>(complexSizeOut)};
    std::vector<std::vector<double>> complexInvalidOut = {std::vector<double>(complexSizeOut + 1)};
    std::vector<std::vector<double>> complexOutUnequalSize = {std::vector<double>(complexSizeOut),
                                                              std::vector<double>(complexSizeOut)};
    std::vector<std::vector<double>> complexInputUnequalSize = {std::vector<double>(complexSizeInput)};
    complexNetwork.initializeFitting(std::make_shared<nn::losses::MSE>());

    ASSERT_THROW(complexNetwork.fit(complexInvalidInput, complexValidOut, 1.0, 1),
                 nn::Backpropagator::InvalidVectorSize);

    ASSERT_THROW(complexNetwork.fit(complexValidInput, complexInvalidOut, 1.0, 1),
                 nn::Backpropagator::InvalidVectorSize);

    ASSERT_THROW(complexNetwork.fit(complexInvalidInput, complexInvalidOut, 1.0, 1),
                 nn::Backpropagator::InvalidVectorSize);

    ASSERT_THROW(complexNetwork.fit(complexInputUnequalSize, complexOutUnequalSize, 1.0, 1),
                 nn::Backpropagator::InvalidVectorSize);

    ASSERT_NO_THROW(complexNetwork.fit(complexValidInput, complexValidOut, 1.0, 1));
}

TEST_F(Backpropagator, Fit_WillThrowWhenBackpropagatorIsntIntialized)
{
    ASSERT_THROW(simpleNetwork.fit({std::vector<double>(simpleNetwork.getInputLayerSize())},
                                   {std::vector<double>(simpleNetwork.getOutputLayerSize())}, 1.0, 1),
                 nn::Backpropagator::NotInitializedException);

    simpleNetwork.initializeFitting(std::make_shared<nn::losses::MSE>());
    ASSERT_NO_THROW(simpleNetwork.fit({std::vector<double>(simpleNetwork.getInputLayerSize())},
                                      {std::vector<double>(simpleNetwork.getOutputLayerSize())}, 1.0, 1));

    ASSERT_NO_THROW(simpleNetwork.fit({std::vector<double>(simpleNetwork.getInputLayerSize())},
                                      {std::vector<double>(simpleNetwork.getOutputLayerSize())}, 1.0, 1));
}

TEST_F(Backpropagator, Fit_WillThrowWhenBatchSizeIsInvalid)
{
    simpleNetwork.initializeFitting(std::make_shared<nn::losses::MSE>());
    ASSERT_THROW(simpleNetwork.fit({std::vector<double>(simpleNetwork.getInputLayerSize())},
                                   {std::vector<double>(simpleNetwork.getOutputLayerSize())}, 1.0, 1, -1),
                 nn::Backpropagator::InvalidBatchSize);

    ASSERT_THROW(simpleNetwork.fit({std::vector<double>(simpleNetwork.getInputLayerSize())},
                                   {std::vector<double>(simpleNetwork.getOutputLayerSize())}, 1.0, 1, 0),
                 nn::Backpropagator::InvalidBatchSize);

    ASSERT_THROW(simpleNetwork.fit({std::vector<double>(simpleNetwork.getInputLayerSize())},
                                   {std::vector<double>(simpleNetwork.getOutputLayerSize())}, 1.0, 1, 2),
                 nn::Backpropagator::InvalidBatchSize);

    ASSERT_NO_THROW(simpleNetwork.fit({std::vector<double>(simpleNetwork.getInputLayerSize())},
                                      {std::vector<double>(simpleNetwork.getOutputLayerSize())}, 1.0, 1, 1));
}

TEST_F(Backpropagator, Fit_WillThrowWhenEpochCountIsInvalid)
{
    simpleNetwork.initializeFitting(std::make_shared<nn::losses::MSE>());
    ASSERT_THROW(simpleNetwork.fit({std::vector<double>(simpleNetwork.getInputLayerSize())},
                                   {std::vector<double>(simpleNetwork.getOutputLayerSize())}, 1.0, -1),
                 nn::Backpropagator::InvalidEpochCount);

    ASSERT_NO_THROW(simpleNetwork.fit({std::vector<double>(simpleNetwork.getInputLayerSize())},
                                      {std::vector<double>(simpleNetwork.getOutputLayerSize())}, 1.0, 1));

    ASSERT_NO_THROW(simpleNetwork.fit({std::vector<double>(simpleNetwork.getInputLayerSize())},
                                      {std::vector<double>(simpleNetwork.getOutputLayerSize())}, 1.0, 0));
}

TEST_F(Backpropagator, Fit_WillGradientBeCallculatedCorrectlySimpleNetwork)
{
    std::vector<std::vector<double>> xs = {{0.5},
                                           {1.0},
                                           {1.5}};
    std::vector<std::vector<double>> ys = {{1.0},
                                           {2.0},
                                           {3.0}};
    double learningRate = 1e-4;

    simpleNetwork.initializeFitting(std::make_shared<nn::losses::MSE>());
    simpleNetwork.fit(xs, ys, learningRate, 1);

    double biases[] = {4.9558125, 4.9778125, 4.99105};
    double weights[] = {0.0, 1.8668125, 2.34775};

    for (int l = 0; l < simpleNetwork.getNumberOfLayers(); l++)
    {
        EXPECT_EQ(simpleNetwork.getLayer(l)->getNeuron(0)->getB(), biases[l]);
        if (l > 0)
            EXPECT_EQ(simpleNetwork.getLayer(l)->getNeuron(0)->getConnectionPreviousLayer(simpleNetwork.getLayer(
                    l - 1)->getNeuron(0).get())->w, weights[l]);
    }
}