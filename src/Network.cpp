#include "Network.h"

nn::Network::Network(int initialNumberOfLayers) : backpropagator(
        std::make_shared<nn::Backpropagator>())
{
    layers.reserve(initialNumberOfLayers);
}

int nn::Network::getNumberOfLayers() const
{
    return size;
}

void nn::Network::pushLayer(const std::shared_ptr<nn::abs::Layer>& l)
{
    if (size == 0 && !isInputLayer(l))
        throw InvalidFirstLayerException("The first layer that is being pushed must derive from nn::abs::InputLayer");

    if (size != 0 && isInputLayer(l))
        std::cout << "[nn::Network::pushLayer] WARNING: The layer that is being pushed is of type nn::abs::InputLayer, "
                     "that means the values before that layer can't be passed through this layer";

    if (size > 0)
        layers[size - 1]->connect(l.get());
    layers.push_back(l);
    size++;
}

bool nn::Network::isInputLayer(
        const std::shared_ptr<nn::abs::Layer>& l) const { return dynamic_cast<nn::abs::InputLayer*>(l.get()); }

int nn::Network::getCapacityOfLayers() const
{
    return layers.capacity();
}

nn::Network::Network(int initialNumberOfLayers, const std::shared_ptr<nn::abs::InputLayer>& firstLayer) : layers(
        initialNumberOfLayers),
                                                                                                          backpropagator(
                                                                                                                  std::dynamic_pointer_cast<nn::abs::Backpropagator>(
                                                                                                                          std::make_shared<nn::Backpropagator>()))
{
    nn::Network::pushLayer(firstLayer);
}

nn::Network::Network(const std::shared_ptr<nn::abs::InputLayer>& firstLayer) : backpropagator(
        std::make_shared<nn::Backpropagator>())
{
    nn::Network::pushLayer(firstLayer);
}

nn::Network::Network(const std::shared_ptr<nn::abs::InputLayer>& firstLayer,
                     const std::vector<std::shared_ptr<nn::abs::Layer>>& layers) : backpropagator(
        std::make_shared<nn::Backpropagator>())
{
    nn::Network::pushLayer(firstLayer);

    this->layers.insert(this->layers.end(), layers.begin(), layers.end());
    size += layers.size();
}

void nn::Network::setInputs(const std::vector<double>& values)
{
    areLayersGiven();
    dynamic_cast<nn::abs::InputLayer*>(layers[0].get())->setValues(values);
}

std::shared_ptr<const nn::abs::Layer> nn::Network::getLayer(int index) const
{
    return layers[index];
}

void nn::Network::setActivation(int index, const std::shared_ptr<nn::abs::Activation>& f)
{
    layers[index]->setActivation(f);
}

void nn::Network::setBias(int index, std::vector<double> bs)
{
    layers[index]->setBias(bs);
}

void nn::Network::setWeights(int index, const std::vector<std::map<std::shared_ptr<nn::abs::Neuron>, double>>& weights)
{
    layers[index]->setWeights(weights);
}

const std::vector<double>& nn::Network::calculate()
{
    areLayersGiven();
    resetCaches();
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

void nn::Network::resetCaches() const
{
    for (auto& l : layers)
        l->resetCaches();
}

nn::Network::Network() : backpropagator(
        std::make_shared<nn::Backpropagator>())
{

}

void nn::Network::setBackpropagator(const std::shared_ptr<nn::abs::Backpropagator>& b)
{
    backpropagator = std::move(b);
}

void nn::Network::initializeFitting(const std::shared_ptr<nn::abs::LossFunction>& lossF)
{
    backpropagator->initialize(this, lossF);
}

int nn::Network::getOutputLayerSize() const
{
    return getLayer(getNumberOfLayers() - 1)->getSize();
}

int nn::Network::getInputLayerSize() const
{
    return getLayer(0)->getSize();
}

const std::shared_ptr<nn::abs::Layer>& nn::Network::getLayer(int index)
{
    return layers[index];
}

void nn::Network::fit(const std::vector<std::vector<double>>& x, const std::vector<std::vector<double>>& y, double learningRate, int epochs,
                      int batchSize)
{
    backpropagator->fit(x, y, learningRate, epochs, batchSize);
}

void nn::Network::fit(const std::vector<std::vector<double>>& x, const std::vector<std::vector<double>>& y, double learingRate, int epochs)
{
    backpropagator->fit(x, y, learingRate, epochs);
}

const std::vector<double>& nn::Network::calculate(std::vector<double> values)
{
    setInputs(values);
    return calculate();
}
