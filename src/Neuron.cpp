#include "Neuron.h"

nn::Neuron::Neuron(std::vector<nn::Connection*> connectionsNextLayer,
                   std::vector<nn::Connection*> connectionsPreviousLayer) {}

std::vector<std::shared_ptr<nn::abs::Connection>> nn::Neuron::getConnectionsNextLayer()
{
    return connectionsNextLayer;
}

std::vector<std::shared_ptr<nn::abs::Connection>> nn::Neuron::getConnectionsPreviousLayer()
{
    return connectionsPreviousLayer;
}

void nn::Neuron::connect(nn::abs::Neuron* n)
{
    connectionsNextLayer.push_back(std::make_shared<Connection>(Connection{this, n}));
}

nn::Connection::Connection(nn::abs::Neuron* from, nn::abs::Neuron* to) : from(from), to(to) {}