#include "test/Neuron.h"

using namespace nn::test;

void Neuron::SetUp()
{
    n = nn::Neuron{connectionsNextLayer, connectionsPreviousLayer};
}

TEST_F(Neuron, ConstructorTest_WillAllWeightsAndBaiosesBeZero)
{
    for (auto& c : n.getConnectionsNextLayer())
    {
        ASSERT_EQ(c->w, 0.0);
        ASSERT_EQ(c->b, 0.0);
        ASSERT_EQ((unsigned long) ((nn::Connection*) c.get())->from, 0l);
        ASSERT_EQ((unsigned long) ((nn::Connection*) c.get())->to, 0l);
    }

    for (auto c : n.getConnectionsPreviousLayer())
    {
        ASSERT_EQ(c->w, 0.0);
        ASSERT_EQ(c->b, 0.0);
        ASSERT_EQ((unsigned long) ((nn::Connection*) c.get())->from, 0l);
        ASSERT_EQ((unsigned long) ((nn::Connection*) c.get())->to, 0l);
    }
}

TEST_F(Neuron, ConnectionConnection_WillThisBeInConnectionsPreviousLayerOfNextLayerNeuron) {
    nn::Neuron n2{};
    n.connect(&n2);

    bool isN2InConnectionsNextLayer = false;
    for (auto& c : n.getConnectionsNextLayer())
        isN2InConnectionsNextLayer |= ((nn::Connection*) c.get())->to == &n2;
    ASSERT_TRUE(isN2InConnectionsNextLayer);
}