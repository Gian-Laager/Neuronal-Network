#include "Neuron.h"

#include <utility>

nn::Neuron::Neuron(
        std::map<nn::abs::Neuron*, std::shared_ptr<nn::abs::Connection>> connectionsNextLayer,
        std::map<nn::abs::Neuron*, std::shared_ptr<nn::abs::Connection>> connectionsPreviousLayer)
        : connectionsNextLayer(std::move(connectionsNextLayer)),
          connectionsPreviousLayer(std::move(connectionsPreviousLayer)) {}

const std::map<nn::abs::Neuron*, std::shared_ptr<nn::abs::Connection>>&
nn::Neuron::getConnectionsNextLayer()
{
    return connectionsNextLayer;
}

const std::map<nn::abs::Neuron*, std::shared_ptr<nn::abs::Connection>>&
nn::Neuron::getConnectionsPreviousLayer()
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
    if (!cacheSet)
    {
        cacheActivation = (*activationFunction)(getZ());
        cacheSet = true;
    }
    return cacheActivation;
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

void nn::Neuron::setActivation(const std::shared_ptr<nn::abs::Activation>& f)
{
    activationFunction = f;
}

void nn::Neuron::setB(double bias)
{
    this->b = bias;
}

void nn::Neuron::setWeights(const std::map<nn::abs::Neuron*, double>& weights)
{
    isInValidKeyInWeights(weights);

    for (auto& w : weights)
        connectionsNextLayer[w.first]->w = w.second;
}

void nn::Neuron::isInValidKeyInWeights(const std::map<nn::abs::Neuron*, double>& weights) const
{
    for (auto& w : weights)
        if (this->connectionsNextLayer.find(w.first) == this->connectionsNextLayer.end())
            throw nn::Neuron::InvalidKeyInMapException(
                    "All keys of the weights map have to be in connectionsNextLayer");
}

void nn::Neuron::appendToPreviousConnection(nn::abs::Neuron* n,
                                            const std::shared_ptr<nn::abs::Connection>& c)
{
    connectionsPreviousLayer[n] = c;
}

void nn::Neuron::resetCache() const
{
    cacheSet = false;
    cacheZSet = false;
}

std::shared_ptr<const nn::abs::Activation> nn::Neuron::getActivation() const
{
    return activationFunction;
}

double nn::Neuron::getZ() const
{
    if (!cacheZSet)
    {
        cacheZ = multiplyPreviousLayersResultsByWeights() + b;
        cacheZSet = true;
    }

    return cacheZ;
}

const std::shared_ptr<nn::abs::Connection>&
nn::Neuron::getConnectionNextLayer(nn::abs::Neuron* index)
{
    return connectionsNextLayer[index];
}

const std::shared_ptr<nn::abs::Connection>&
nn::Neuron::getConnectionPreviousLayer(nn::abs::Neuron* index)
{
    return connectionsPreviousLayer[index];
}

const std::shared_ptr<nn::abs::Activation>& nn::Neuron::getActivation()
{
    return activationFunction;
}

nn::Connection::Connection(nn::abs::Neuron* from, nn::abs::Neuron* to) : from(from),
                                                                         to(to) {}


nn::InputNeuron::InputNeuron(
        std::map<nn::abs::Neuron*, std::shared_ptr<nn::abs::Connection>> connectionsNextLayer)
        : connectionsNextLayer(std::move(
        connectionsNextLayer), {}) {}

double nn::InputNeuron::getValue() const
{
    if (!cacheSet)
    {
        cacheActivation = (*activationFunction)(getZ());
        cacheSet = true;
    }

    return cacheActivation;
}

void nn::InputNeuron::setValue(double v)
{
    value = v;
}

nn::InputNeuron::InputNeuron(double v,
                             std::map<nn::abs::Neuron*, std::shared_ptr<nn::abs::Connection>> connectionsNextLayer)
        : connectionsNextLayer(std::move(connectionsNextLayer)),
          value(v)
{

}

nn::InputNeuron::InputNeuron(double v)
        : connectionsNextLayer(),
          value(v)
{

}

const std::map<nn::abs::Neuron*, std::shared_ptr<nn::abs::Connection>>&
nn::InputNeuron::getConnectionsNextLayer()
{
    return connectionsNextLayer;
}

void nn::InputNeuron::connect(nn::abs::Neuron* n)
{
    auto c = std::make_shared<Connection>(this, n);
    connectionsNextLayer[n] = c;
    n->appendToPreviousConnection(this, c);
}

const std::map<nn::abs::Neuron*, std::shared_ptr<nn::abs::Connection>>&
nn::InputNeuron::getConnectionsPreviousLayer()
{
    return std::move(std::map<nn::abs::Neuron*, std::shared_ptr<nn::abs::Connection>>{});
}

double nn::InputNeuron::getB() const
{
    return b;
}

void nn::InputNeuron::setB(double bias)
{
    this->b = bias;
}

void nn::InputNeuron::setActivation(const std::shared_ptr<nn::abs::Activation>& f)
{
    activationFunction = f;
}

void nn::InputNeuron::setWeights(const std::map<nn::abs::Neuron*, double>& weights)
{
    isInValidKeyInWeights(weights);

    for (auto& w : weights)
        connectionsNextLayer[w.first]->w = w.second;
}

void nn::InputNeuron::isInValidKeyInWeights(const std::map<nn::abs::Neuron*, double>& weights) const
{
    for (auto& w : weights)
        if (connectionsNextLayer.find(w.first) == connectionsNextLayer.end())
            throw nn::Neuron::InvalidKeyInMapException(
                    "All keys of the weights map have to be in connectionsNextLayer");
}

void nn::InputNeuron::appendToPreviousConnection(nn::abs::Neuron* n,
                                                 const std::shared_ptr<nn::abs::Connection>& c)
{

}

void nn::InputNeuron::resetCache() const
{
    cacheSet = false;
    cacheZSet = false;
}

std::shared_ptr<const nn::abs::Activation> nn::InputNeuron::getActivation() const
{
    return activationFunction;
}

double nn::InputNeuron::getZ() const
{
    if (!cacheZSet)
    {
        cacheZ = value + b;
        cacheZSet = true;
    }
    return cacheZ;
}

const std::shared_ptr<nn::abs::Connection>&
nn::InputNeuron::getConnectionNextLayer(nn::abs::Neuron* index)
{
    return connectionsNextLayer[index];
}

const std::shared_ptr<nn::abs::Connection>& nn::InputNeuron::getConnectionPreviousLayer(
        nn::abs::Neuron* index)
{
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wreturn-stack-address"
    return std::make_shared<nn::Connection>(nullptr, this);
#pragma clang diagnostic pop
}

const std::shared_ptr<nn::abs::Activation>& nn::InputNeuron::getActivation()
{
    return activationFunction;
}
