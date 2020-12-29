#include "test/sycl/Layer.h"

using nn::sycl::test::Sycl_Layer;

void nn::sycl::test::Sycl_Layer::SetUp()
{
    layer = nn::sycl::Layer{numberOfNeurons};
}

TEST_PF(Sycl, Layer, Constructor_WillSizeBeSetRight)
{
    ASSERT_EQ(layer.getSize(), numberOfNeurons);
}

TEST_PF(Sycl, Layer, GetNeurons_WillReturnedVectorHaveSameSizeAsNetwork)
{
    ASSERT_EQ(layer.getNeurons().size(), numberOfNeurons);
}

TEST_PF(Sycl, Layer, GetNeurons_WillReturnedVectorHaveSameSizeAsNetworkConst)
{
    const nn::sycl::Layer cLayer = nn::sycl::Layer(numberOfNeurons);
    ASSERT_EQ(cLayer.getNeurons().size(), numberOfNeurons);
}

TEST_PF(Sycl, Layer, SetBias_WillGetNeuronsHaveRightBiasValues)
{
    std::vector<double> biases = {
            1.0, 2.0, 3.0
    };
    layer.setBias(biases);
    std::vector<std::shared_ptr<nn::abs::Neuron>> neurons = layer.getNeurons();
    for (int i = 0; i < layer.getSize(); i++)
        ASSERT_EQ(neurons[i]->getB(), biases[i]);
}

TEST_PF(Sycl, Layer, SetBias_WillSetBiasThrowIfInvalidVectorSizeIsGiven)
{
    std::vector<double> bsInvalid(numberOfNeurons + 1);
    std::vector<double> bsValid(numberOfNeurons);

    ASSERT_THROW(layer.setBias(bsInvalid), nn::sycl::Layer::IncompatibleVectorException);
    ASSERT_NO_THROW(layer.setBias(bsValid));
}

TEST_PF(Sycl, Layer, SetActivation_WillGetNeuronsHaveRightActivatoin)
{
    std::shared_ptr<nn::abs::Activation> activationPtr = std::make_shared<nn::activations::Sigmoid>();
    layer.setActivation(activationPtr);

    std::vector<std::shared_ptr<nn::abs::Neuron>> neurons = layer.getNeurons();
    double x = 100.0;
    for (int i = 0; i < layer.getSize(); i++)
    {
        ASSERT_EQ((*neurons[i]->getActivation())(x), (*activationPtr)(x));
        ASSERT_EQ(neurons[i]->getActivation()->derivative(x), activationPtr->derivative(x));
    }
}

TEST_PF(Sycl, Layer, ConstructorWithActivation_WillGetNeuronsHaveRightActivatoin)
{
    std::shared_ptr<nn::abs::Activation> activationPtr = std::make_shared<nn::activations::Sigmoid>();
    nn::sycl::Layer l{numberOfNeurons, activationPtr};

    std::vector<std::shared_ptr<nn::abs::Neuron>> neurons = l.getNeurons();
    double x = 100.0;
    for (int i = 0; i < l.getSize(); i++)
    {
        ASSERT_EQ((*neurons[i]->getActivation())(x), (*activationPtr)(x));
        ASSERT_EQ(neurons[i]->getActivation()->derivative(x), activationPtr->derivative(x));
    }
}

TEST_PF(Sycl, Layer, GetNeuron_WillBeEqualToGetNeuronsWithIndex)
{
    std::shared_ptr<nn::abs::Activation> activationPtr = std::make_shared<nn::activations::Sigmoid>();
    layer.setActivation(activationPtr);

    std::vector<double> biases = {
            1.0, 2.0, 3.0
    };
    layer.setBias(biases);
    std::vector<std::shared_ptr<nn::abs::Neuron>> neurons = layer.getNeurons();

    double x = 100.0;
    for (int i = 0; i < layer.getSize(); i++)
    {
        ASSERT_EQ(neurons[i]->getB(), layer.getNeuron(i)->getB());
        ASSERT_EQ((*neurons[i]->getActivation())(x), (*layer.getNeuron(i)->getActivation())(x));
        ASSERT_EQ(neurons[i]->getActivation()->derivative(x), layer.getNeuron(i)->getActivation()->derivative(x));
    }
}

TEST_PF(Sycl, Layer, ConnectToSyclLayer_WillGetNeuronHaveConnections)
{
    std::shared_ptr<nn::sycl::Layer> l1 = std::make_shared<nn::sycl::Layer>(3);
    std::shared_ptr<nn::sycl::Layer> l2 = std::make_shared<nn::sycl::Layer>(2);

    l1->connect(std::static_pointer_cast<nn::sycl::abs::Layer>(l2));

    for (auto& nInL1 : l1->getNeurons())
    {
        bool connectedToAll = true;
        if (nInL1->getConnectionsNextLayer().size() == 0)
            connectedToAll = false;
        for (auto& connection : nInL1->getConnectionsNextLayer())
        {
            bool connected = false;
            for (auto& nInL2 : l2->getNeurons())
                connected |= (long) connection.first == (long) nInL2.get();
            connectedToAll &= connected;
        }
        ASSERT_TRUE(connectedToAll);
    }

    for (auto& nInL2 : l2->getNeurons())
    {
        bool connectedToAll = true;
        if (nInL2->getConnectionsPreviousLayer().size() == 0)
            connectedToAll = false;
        for (auto& connection : nInL2->getConnectionsPreviousLayer())
        {
            bool connected = false;
            for (auto& nInL1 : l1->getNeurons())
                connected |= (long) connection.first == (long) nInL1.get();
            connectedToAll &= connected;
        }
        ASSERT_TRUE(connectedToAll);
    }
}

TEST_PF(Sycl, Layer, ConnectToNormalLayer_WillGetNeuronHaveConnections)
{
    std::shared_ptr<nn::sycl::Layer> l1 = std::make_shared<nn::sycl::Layer>(3);
    std::shared_ptr<nn::Layer<nn::Neuron>> l2 = std::make_shared<nn::Layer<nn::Neuron>>(2);

    l1->connect(l2);

    for (auto& nInL1 : l1->getNeurons())
    {
        bool connectedToAll = true;
        if (nInL1->getConnectionsNextLayer().size() == 0)
            connectedToAll = false;
        for (auto& connection : nInL1->getConnectionsNextLayer())
        {
            bool connected = false;
            for (auto& nInL2 : l2->getNeurons())
                connected |= (long) connection.first == (long) nInL2.get();
            connectedToAll &= connected;
        }
        ASSERT_TRUE(connectedToAll);
    }

    for (auto& nInL2 : l2->getNeurons())
    {
        bool connectedToAll = true;
        if (nInL2->getConnectionsPreviousLayer().size() == 0)
            connectedToAll = false;
        for (auto& connection : nInL2->getConnectionsPreviousLayer())
        {
            bool connected = false;
            for (auto& nInL1 : l1->getNeurons())
                connected |= (long) connection.first == (long) nInL1.get();
            connectedToAll &= connected;
        }
        ASSERT_TRUE(connectedToAll);
    }
}

TEST_PF(Sycl, Layer, InputLayer_SetValue_WillCalculateReturnRightOutPut)
{
    nn::sycl::InputLayer inLayer{4};
    std::vector<double> biases = {1.0, 2.0, 3.0, 4.0};
    std::shared_ptr<nn::abs::Activation> activation = std::make_shared<nn::activations::Sigmoid>();
    inLayer.setBias(biases);
    inLayer.setActivation(activation);

    std::vector<double> inputs = {10.0, 20.0, 30.0, 40.0};
    std::vector<double> expectedOutputs(inLayer.getSize());
    for (int i = 0; i < inLayer.getSize(); i++)
        expectedOutputs[i] = (*activation)(inputs[i] + biases[i]);
    inLayer.setValues(inputs);
    auto output = inLayer.calculate();
    for (int i = 0; i < inLayer.getSize(); i++)
        ASSERT_EQ(output[i], expectedOutputs[i]);
}

TEST_PF(Sycl, Layer, MultipleLayer_Calculate_WillOutPutRightValues)
{
    for (int j = 0; j < 1000; j++)
    {
        class FirstLayerActivation : public nn::abs::Activation
        {
        public:
            double operator()(double z) const override { return z - 1; }

            double derivative(double z) const override { return 1; }
        };
        class SecondLayerActivation : public nn::abs::Activation
        {
        public:
            double operator()(double z) const override { return 1 / z; }

            double derivative(double z) const override { return 1; }
        };
        class ThirdLayerActivation : public nn::abs::Activation
        {
        public:
            double operator()(double z) const override { return z * z; }

            double derivative(double z) const override { return 1.0; }
        };

        std::shared_ptr<nn::sycl::InputLayer> inputLayer = std::make_shared<nn::sycl::InputLayer>(2,
                                                                                                  std::make_shared<FirstLayerActivation>());
        std::shared_ptr<nn::sycl::abs::Layer> layer1 = std::make_shared<nn::sycl::Layer>(3,
                                                                                         std::make_shared<SecondLayerActivation>());
        std::shared_ptr<nn::sycl::abs::Layer> layer2 = std::make_shared<nn::sycl::Layer>(2,
                                                                                         std::make_shared<ThirdLayerActivation>());

        inputLayer->connect(layer1);
        layer1->connect(layer2);

        inputLayer->setBias(std::vector<double>{0.2, -3});
        std::vector<nn::abs::Connection> w = {{0.75},
                                              {0.3},
                                              {0.5},
                                              {0.2},
                                              {0.25},
                                              {0.1}};
        inputLayer->setWeights(cl::sycl::buffer<nn::abs::Connection, 2>{w.data(), cl::sycl::range<2>{
                (unsigned long) layer1->getSize(), (unsigned long) inputLayer->getSize()}});
        layer1->setBias(std::vector<double>{0.8, 0.3, 0.2});
        std::vector<nn::abs::Connection> w2 = {{0.2},
                                               {0.6},
                                               {1.0},
                                               {0.4},
                                               {0.8},
                                               {1.2}};

        layer1->setWeights(cl::sycl::buffer<nn::abs::Connection, 2>{w2.data(),
                                                                    cl::sycl::range<2>{
                                                                            (unsigned long) layer2->getSize(),
                                                                            (unsigned long) layer1->getSize()}});
        layer2->setBias(std::vector<double>{-1.2, -0.1});

        std::vector<double> inputs = {0.5, 0.25};
        inputLayer->setValues(inputs);
        std::vector<double> resultCalc = layer2->calculate();
        std::vector<double> resultRight = {130321.0 / 3025.0, 48.450036730945811};

        ASSERT_EQ(resultCalc.size(), layer2->getSize());
        ASSERT_EQ(resultCalc.size(), resultRight.size());

        for (int i = 0; i < resultRight.size(); i++)
            ASSERT_EQ(resultCalc[i], resultRight[i]) << j;
    }
}

TEST_PF(Sycl, Layer, CalculateSycl_InputLayer_WillReturnRightOutut)
{
    nn::sycl::InputLayer inLayer{4};
    std::vector<double> biases = {1.0, 2.0, 3.0, 4.0};
    std::shared_ptr<nn::abs::Activation> activation = std::make_shared<nn::activations::Sigmoid>();
    inLayer.setBias(biases);
    inLayer.setActivation(activation);

    std::vector<double> inputs = {10.0, 20.0, 30.0, 40.0};
    std::vector<double> expectedOutputs(inLayer.getSize());
    for (int i = 0; i < inLayer.getSize(); i++)
        expectedOutputs[i] = (*activation)(inputs[i] + biases[i]);

    inLayer.setValues(inputs);
    cl::sycl::buffer<double, 1> output;
    output = inLayer.calculateSycl();

    for (int i = 0; i < inLayer.getSize(); i++)
        ASSERT_EQ(output.get_access<cl::sycl::access::mode::read>()[i], expectedOutputs[i]);
}

TEST_PF(Sycl, Layer, CalculateSycl_InputLayerAndLayer_WillReturnRightOutut)
{
    std::shared_ptr<nn::sycl::InputLayer> inLayer = std::make_shared<nn::sycl::InputLayer>(4);
    std::shared_ptr<nn::sycl::abs::Layer> layer = std::make_shared<nn::sycl::Layer>(2);
    inLayer->connect(layer);

    std::vector<double> biases = {1.0, 2.0, 3.0, 4.0};
    std::shared_ptr<nn::abs::Activation> activation = std::make_shared<nn::activations::Sigmoid>();
    inLayer->setBias(biases);
    inLayer->setActivation(activation);
    std::vector<nn::abs::Connection> weights = {
            {1.0},
            {2.0},
            {2.0},
            {1.0},
            {1.0},
            {2.0},
            {2.0},
            {1.0}
    };
    inLayer->setWeights(cl::sycl::buffer<nn::abs::Connection, 2>{weights.data(), cl::sycl::range<2>{2, 4}});
    std::vector<double> biasLayer = {
            1.0, 2.0
    };
    layer->setBias(biasLayer);
}

TEST_PF(Sycl, Layer, SetWeights_2dVectors_WillSetWeightsRight)
{
    for (int j = 0; j < 1000; j++)
    {
        class FirstLayerActivation : public nn::abs::Activation
        {
        public:
            double operator()(double z) const override { return z - 1; }

            double derivative(double z) const override { return 1; }
        };

        class SecondLayerActivation : public nn::abs::Activation
        {
        public:
            double operator()(double z) const override { return 1 / z; }

            double derivative(double z) const override { return 1; }
        };

        class ThirdLayerActivation : public nn::abs::Activation
        {
        public:
            double operator()(double z) const override { return z * z; }

            double derivative(double z) const override { return 1.0; }
        };

        std::shared_ptr<nn::sycl::InputLayer> inputLayer = std::make_shared<nn::sycl::InputLayer>(2,
                                                                                                  std::make_shared<FirstLayerActivation>());
        std::shared_ptr<nn::sycl::abs::Layer> layer1 = std::make_shared<nn::sycl::Layer>(3,
                                                                                         std::make_shared<SecondLayerActivation>());
        std::shared_ptr<nn::sycl::abs::Layer> layer2 = std::make_shared<nn::sycl::Layer>(2,
                                                                                         std::make_shared<ThirdLayerActivation>());

        inputLayer->connect(layer1);
        layer1->connect(layer2);

        inputLayer->setBias(std::vector<double>{0.2, -3}
        );
        std::vector<std::vector<nn::abs::Connection>> w = {{{0.75}, {0.3}},
                                                           {{0.5},  {0.2}},
                                                           {{0.25}, {0.1}}};
        inputLayer->setWeights(w);
        layer1->setBias(std::vector<double>{0.8, 0.3, 0.2});
        std::vector<std::vector<nn::abs::Connection>> w2 = {{{0.2}, {0.6}, {1.0},},
                                                            {{0.4}, {0.8}, {1.2}}};

        layer1->setWeights(w2);
        layer2->setBias(std::vector<double>{-1.2, -0.1}
        );

        std::vector<double> inputs = {0.5, 0.25};
        inputLayer->setValues(inputs);
        std::vector<double> resultCalc = layer2->calculate();
        std::vector<double> resultRight = {130321.0 / 3025.0, 48.450036730945811};

        ASSERT_EQ(resultCalc.size(), layer2->getSize());
        ASSERT_EQ(resultCalc.size(), resultRight.size());

        for (int i = 0; i < resultRight.size(); i++)
            ASSERT_EQ(resultCalc[i], resultRight[i]) << j;
    }
}