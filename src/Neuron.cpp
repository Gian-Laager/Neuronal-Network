#include "Neuron.h"

#include <utility>

nn::Neuron::Neuron(std::map<nn::abs::Neuron*, std::shared_ptr<nn::abs::Connection>> connectionsNextLayer,
                   std::map<nn::abs::Neuron*, std::shared_ptr<nn::abs::Connection>> connectionsPreviousLayer) {}

std::map<nn::abs::Neuron*, std::shared_ptr<nn::abs::Connection>> nn::Neuron::getConnectionsNextLayer()
{
    return connectionsNextLayer;
}

std::map<nn::abs::Neuron*, std::shared_ptr<nn::abs::Connection>> nn::Neuron::getConnectionsPreviousLayer()
{
    return connectionsPreviousLayer;
}

void nn::Neuron::connect(nn::abs::Neuron* n)
{
    auto c = std::make_shared<Connection>(Connection{this, n});
    connectionsNextLayer[n] = c;
    ((nn::Neuron*) n)->connectionsPreviousLayer[this] = c;
}

double nn::Neuron::getValue() const
{
    double result = 0.0;
    for (auto& connection : connectionsPreviousLayer)
        result += ((nn::Connection*) connection.second.get())->from->getValue() * connection.second->w;
    result += b;
    return activationFunction(result);
}

double nn::Neuron::getB() const
{
    return b;
}

void nn::Neuron::setActivation(std::function<double(double)> f)
{
    activationFunction = std::move(f);
}

void nn::Neuron::setB(double b)
{
    this->b = b;
}

nn::Connection::Connection(nn::abs::Neuron* from, nn::abs::Neuron* to) : from(from),
                                                                         to(to) {}


nn::BeginNeuron::BeginNeuron(std::map<nn::abs::Neuron*, std::shared_ptr<nn::abs::Connection>> connectionsNextLayer)
        : nn::Neuron(std::move(
        connectionsNextLayer), {}) {}

double nn::BeginNeuron::getValue() const
{
    return activationFunction(value + b);
}

void nn::BeginNeuron::setValue(double v)
{
    value = v;
}

nn::BeginNeuron::BeginNeuron(double v,
                             std::map<nn::abs::Neuron*, std::shared_ptr<nn::abs::Connection>> connectionsNextLayer)
        : nn::Neuron(connectionsNextLayer, {}),
          value(v)
{

}

nn::BeginNeuron::BeginNeuron(double v)
        : nn::Neuron(),
          value(v)
{

}
