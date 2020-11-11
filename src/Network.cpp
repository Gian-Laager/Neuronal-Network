#include "Network.h"

nn::Network::Network(int initialNumberOfLayers)
{
    layers.reserve(initialNumberOfLayers);
}

int nn::Network::getNumberOfLayers() const
{
    return size;
}

void nn::Network::pushLayer(std::shared_ptr<nn::abs::Layer> l)
{
    if (size == 0 && !isBeginLayer(l))
        throw InvalidFirstLayerException("The first layer that is being pushed must derive from nn::abs::BeginLayer");

    if (size != 0 && isBeginLayer(l))
        std::cout << "[nn::Network::pushLayer] WARNING: The layer that is being pushed is of type nn::abs::BeginLayer, "
                     "that means the values before that layer can't be passed through this layer";

    if (size > 0)
        layers[size - 1]->connect(l.get());
    layers.push_back(l);
    size++;
}

bool nn::Network::isBeginLayer(const std::shared_ptr<nn::abs::Layer>& l) const { return dynamic_cast<nn::abs::BeginLayer*>(l.get()); }

int nn::Network::getCapacityOfLayers() const
{
    return layers.capacity();
}

nn::Network::Network(int initialNumberOfLayers, const std::shared_ptr<nn::abs::BeginLayer>& firstLayer) : layers(
        initialNumberOfLayers)
{
    nn::Network::pushLayer(firstLayer);
}

nn::Network::Network(std::shared_ptr<nn::abs::BeginLayer> firstLayer)
{
    nn::Network::pushLayer(firstLayer);
}

nn::Network::Network(const std::shared_ptr<nn::abs::BeginLayer>& firstLayer, std::vector<std::shared_ptr<nn::abs::Layer>> layers)
{
    nn::Network::pushLayer(firstLayer);

    this->layers.insert(this->layers.end(), layers.begin(), layers.end());
    size += layers.size();
}

void nn::Network::setInputs(std::vector<double> values)
{
    areLayersGiven();
    dynamic_cast<nn::abs::BeginLayer*>(layers[0].get())->setValues(values);
}

std::shared_ptr<nn::abs::Layer> nn::Network::getLayer(int index)
{
    return layers[index];
}

void nn::Network::setActivation(int index, std::function<double(double)> f)
{
    layers[index]->setActivation(f);
}

void nn::Network::setBias(int index, std::vector<double> bs)
{
    layers[index]->setBias(bs);
}

void nn::Network::setWeights(int index, const std::vector<std::map<nn::abs::Neuron*, double>>& weights)
{
    layers[index]->setWeights(weights);
}

std::vector<double> nn::Network::calculate() const
{
    areLayersGiven();
    return layers[size - 1]->calculate();
}

void nn::Network::areLayersGiven() const
{
    if (this->size <= 0)
        throw nn::Network::NoLayersGivenException{"No layers were pushed, use pushLayer to add a layer"};
}

int nn::Network::getSize() const
{
    return layers.size();
}
