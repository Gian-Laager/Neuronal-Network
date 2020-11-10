#include "test/Layer.h"

using namespace nn::test;

TEST_F(Layer, Constructor_WillNumberOfNeuronsSetRightNeuronsSize)
{
    ASSERT_EQ(layer.getSize(), numberOfNeurons);
}

TEST_F(Layer, Connect_WillEveryNeuronBeConnectedToTheNeuronsInTheNextLayer)
{
    nn::Layer<nn::Neuron> l2{8};
    layer.connect(&l2);

    for (auto& nInLayer : layer.getNeurons())
    {
        bool connectedToAll = true;
        for (auto& connection : nInLayer->getConnectionsNextLayer())
        {
            bool connected = true;
            for (auto& nInL2 : l2.getNeurons())
                connected |= (long) ((nn::Connection*) connection.second.get())->to == (long) &nInLayer;
            connectedToAll &= connected;
        }
        ASSERT_TRUE(connectedToAll);
    }

    for (auto& nInL2 : l2.getNeurons())
    {
        bool connectedToAll = true;
        for (auto& connection : nInL2->getConnectionsNextLayer())
        {
            bool connected = true;
            for (auto& nInLayer : layer.getNeurons())
                connected |= (long) ((nn::Connection*) connection.second.get())->from == (long) &nInLayer;
            connectedToAll &= connected;
        }
        ASSERT_TRUE(connectedToAll);
    }
}