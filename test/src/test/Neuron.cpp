#include <Activations.h>
#include "test/Neuron.h"

using nn::test::Neuron;

void Neuron::SetUp()
{
    n = nn::Neuron{connectionsNextLayer, connectionsPreviousLayer};
}

TEST_F(Neuron, ConstructorTest_WillAllWeightsAndBaiosesBeZero)
{
    ASSERT_EQ(n.getB(), 0.0);
    for (auto& c : n.getConnectionsNextLayer())
    {
        ASSERT_EQ(c.second->w, 0.0);
        ASSERT_EQ((unsigned long) ((nn::Connection*) c.second.get())->from, 0l);
        ASSERT_EQ((unsigned long) ((nn::Connection*) c.second.get())->to, 0l);
    }

    for (auto c : n.getConnectionsPreviousLayer())
    {
        ASSERT_EQ(c.second->w, 0.0);
        ASSERT_EQ((unsigned long) ((nn::Connection*) c.second.get())->from, 0l);
        ASSERT_EQ((unsigned long) ((nn::Connection*) c.second.get())->to, 0l);
    }
}

TEST_F(Neuron, ConnectionConnection_WillThisBeInConnectionsPreviousLayerOfNextLayerNeuron)
{
    nn::Neuron n2{};
    n.connect(&n2);

    bool isN2InConnectionsNextLayer = false;
    for (auto& c : n.getConnectionsNextLayer())
        isN2InConnectionsNextLayer |= ((nn::Connection*) c.second.get())->to == &n2;
    ASSERT_TRUE(isN2InConnectionsNextLayer);

    bool isN2InConnectionsPreviousLayer = false;
    for (auto& c : n2.getConnectionsNextLayer())
        isN2InConnectionsPreviousLayer |= ((nn::Connection*) c.second.get())->to == &n2;
    ASSERT_TRUE(isN2InConnectionsNextLayer);
}

TEST_F(Neuron, GetValue2Neurons_WillReturnRightValue)
{
    double beginValue = 5.0;
    nn::InputNeuron n0{beginValue};
    nn::Neuron n1{};
    n0.connect(&n1);

    double w = 0.5;
    double b = 10.0;
    std::shared_ptr<nn::abs::Activation> activationF = std::make_shared<nn::activations::Linear>();

    n1.setActivation(activationF);
    n0.getConnectionNextLayer(&n1)->w = w;
    n1.setB(b);

    double result = beginValue;
    result *= w;
    result += b;
    result = (*activationF)(result);
    ASSERT_EQ(n1.getValue(), result);
}

TEST_F(Neuron, SetWeights_WillWeightsBeSetRight)
{
    nn::InputNeuron n0{};
    nn::Neuron n2;
    nn::Neuron n1;

    n0.connect(&n1);
    n0.connect(&n2);

    std::map<std::shared_ptr<nn::abs::Neuron>, double> weights = {{&n1, 0.5}, {&n2, 0.3}};
    n0.setWeights(weights);

    ASSERT_EQ(n0.getConnectionNextLayer(&n1)->w, weights[n1]);
    ASSERT_EQ(n0.getConnectionNextLayer(&n2)->w, weights[n2]);

    ASSERT_EQ(n1.getConnectionPreviousLayer(dynamic_cast<nn::abs::Neuron*>(&n0))->w, weights[&n1]);
    ASSERT_EQ(n2.getConnectionPreviousLayer(dynamic_cast<nn::abs::Neuron*>(&n0))->w, weights[&n2]);
}

TEST_F(Neuron, SetB_WillBiasBeSetCorrectly)
{
    double b = 0.25;
    n.setB(b);
    ASSERT_EQ(n.getB(), b);
}