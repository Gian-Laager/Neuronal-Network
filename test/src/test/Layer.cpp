#include "test/Layer.h"

using nn::test::Layer;

TEST_F(Layer, Constructor_WillNumberOfNeuronsSetRightNeuronsSize)
{
    ASSERT_EQ(layer.getSize(), numberOfNeurons);
}

TEST_F(Layer, Connect_WillEveryNeuronBeConnectedToTheNeuronsInTheNextLayer)
{
    nn::Layer<nn::Neuron> l2 = {8};
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

TEST_F(Layer, WillSetActivationSetActivationsOfEveryNeuron)
{
    std::function<double(double)> f = [](double z) -> double { return z; };
    layer.setActivation(f);

    for (auto& n : layer.getNeurons())
        ASSERT_EQ(getActivation((nn::Neuron*) n.get())(5.0), f(5.0));
}

std::function<double(double)> Layer::getActivation(nn::Neuron* n)
{
    return *(std::function<double(double)>*) ((long) n + 64);
}

TEST_F(Layer, SetValues_WillThrowWhenVectorIsInvalidSize)
{
    int size = 4;
    nn::BeginLayer<nn::BeginNeuron> l = {size};

    ASSERT_THROW(l.setValues(std::vector<double>(size + 1)),
                 nn::BeginLayer<nn::BeginNeuron>::IncompatibleVectorException);
    ASSERT_NO_THROW(l.setValues(std::vector<double>(size)));
}

TEST_F(Layer, SetValues_WillSetValuesCorectly)
{
    std::vector<double> values = {1.0, 2.0, 3.0, 4.0};
    nn::BeginLayer<nn::BeginNeuron> l = {(int) values.size()};
    l.setValues(values);

    for (int i = 0; i < l.getSize(); i++)
        ASSERT_EQ(l.getNeurons()[i]->getValue(), values[i]);
}

TEST_F(Layer, GetNeurons_WillReturnRightPointers)
{
    beginLayer.setNeurons(neurons);
    std::vector<std::shared_ptr<nn::abs::Neuron>> getNeurons = beginLayer.getNeurons();
    for (int i = 0; i < getNeurons.size(); i++)
        ASSERT_EQ(getNeurons[i]->getValue(), neurons[i]->getValue());
}

TEST_F(Layer, GetBeginNeurons_WillReturnRightPointers)
{
    beginLayer.setNeurons(neurons);
    std::vector<std::shared_ptr<nn::abs::BeginNeuron>> getNeurons = beginLayer.getBeginNeurons();
    for (int i = 0; i < getNeurons.size(); i++)
        ASSERT_EQ(getNeurons[i]->getValue(), neurons[i]->getValue());
}