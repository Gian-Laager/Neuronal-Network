#include "Neuron.h"

nn::Neuron::Neuron(std::map<nn::abs::Neuron*, std::shared_ptr<nn::abs::Connection>> connectionsNextLayer,
                   std::map<nn::abs::Neuron*, std::shared_ptr<nn::abs::Connection>> connectionsPreviousLayer)
        : connectionsNextLayer(std::move(connectionsNextLayer)),
          connectionsPreviousLayer(std::move(connectionsPreviousLayer)) {}

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
    n->appendToPreviousConnection(this, c);
}

double nn::Neuron::getValue() const
{
    double result = multiplyPreviousLayersResultsByWeights();
    result += b;
    return activationFunction(result);
}

double nn::Neuron::multiplyPreviousLayersResultsByWeights() const
{
    double result = 0.0;
    for (auto& connection : this->connectionsPreviousLayer)
        result += ((nn::Connection*) connection.second.get())->from->getValue() * connection.second->w;
    return result;
}

double nn::Neuron::getB() const
{
    return b;
}

void nn::Neuron::setActivation(std::function<double(double)> f)
{
    activationFunction = std::move(f);
}

void nn::Neuron::setB(double bias)
{
    this->b = bias;
}

void nn::Neuron::setWeights(std::map<nn::abs::Neuron*, double> weights)
{
    isInValidKeyInWeights(weights);

    for (auto& w : weights)
        connectionsNextLayer[w.first]->w = w.second;
}

void nn::Neuron::isInValidKeyInWeights(const std::map<nn::abs::Neuron*, double>& weights) const
{
    for (auto& w : weights)
        if (this->connectionsNextLayer.find(w.first) == this->connectionsNextLayer.end())
            throw nn::Neuron::InvalidKeyInMapException("All keys of the weights map have to be in connectionsNextLayer");
}

void nn::Neuron::appendToPreviousConnection(nn::abs::Neuron* n, std::shared_ptr<nn::abs::Connection> c)
{
    connectionsPreviousLayer[n] = std::move(c);
}

nn::Connection::Connection(nn::abs::Neuron* from, nn::abs::Neuron* to) : from(from),
                                                                         to(to) {}


nn::BeginNeuron::BeginNeuron(std::map<nn::abs::Neuron*, std::shared_ptr<nn::abs::Connection>> connectionsNextLayer)
        : connectionsNextLayer(std::move(
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
        : connectionsNextLayer(std::move(connectionsNextLayer)),
          value(v)
{

}

nn::BeginNeuron::BeginNeuron(double v)
        : connectionsNextLayer(),
          value(v)
{

}

std::map<nn::abs::Neuron*, std::shared_ptr<nn::abs::Connection>> nn::BeginNeuron::getConnectionsNextLayer()
{
    return connectionsNextLayer;
}

void nn::BeginNeuron::connect(nn::abs::Neuron* n)
{
    auto c = std::make_shared<Connection>(Connection{this, n});
    connectionsNextLayer[n] = c;
    n->appendToPreviousConnection(this, c);
}

std::map<nn::abs::Neuron*, std::shared_ptr<nn::abs::Connection>> nn::BeginNeuron::getConnectionsPreviousLayer()
{
    return std::map<nn::abs::Neuron*, std::shared_ptr<nn::abs::Connection>>{};
}

double nn::BeginNeuron::getB() const
{
    return b;
}

void nn::BeginNeuron::setB(double bias)
{
    this->b = bias;
}

void nn::BeginNeuron::setActivation(std::function<double(double)> f)
{
    activationFunction = std::move(f);
}

void nn::BeginNeuron::setWeights(std::map<nn::abs::Neuron*, double> weights)
{
    isInValidKeyInWeights(weights);

    for (auto& w : weights)
        connectionsNextLayer[w.first]->w = w.second;
}

void nn::BeginNeuron::isInValidKeyInWeights(const std::map<nn::abs::Neuron*, double>& weights) const
{
    for (auto& w : weights)
        if (connectionsNextLayer.find(w.first) == connectionsNextLayer.end())
            throw nn::Neuron::InvalidKeyInMapException("All keys of the weights map have to be in connectionsNextLayer");
}

void nn::BeginNeuron::appendToPreviousConnection(nn::abs::Neuron* n, std::shared_ptr<nn::abs::Connection> c)
{

}
