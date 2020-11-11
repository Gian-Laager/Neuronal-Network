#ifndef NEURONAL_NETWORK_LAYER_H
#define NEURONAL_NETWORK_LAYER_H

#include "abstract/Layer.h"

#include <utility>

namespace nn
{
    template<typename NeuronType>
    requires std::is_base_of<nn::abs::Neuron, NeuronType>::value
    class Layer : public nn::abs::Layer
    {
    protected:
        std::vector<std::shared_ptr<nn::abs::Neuron>> neurons;

    public:
        Layer(int numberOfNeurons);

        Layer(std::vector<std::shared_ptr<nn::abs::Neuron>> neurons);

        Layer(int numberOfNeurons, const std::function<double(double)>& f);

        Layer(std::vector<std::shared_ptr<nn::abs::Neuron>> neurons, std::function<double(double)> f);

        Layer() = default;

        int getSize() const override;

        void connect(nn::abs::Layer* l) override;

        std::vector<std::shared_ptr<nn::abs::Neuron>> getNeurons() override;

        void setActivation(std::function<double(double)> f) override;

        void setBias(const std::vector<double>& bs) override;

        void setWeights(const std::vector<std::map<nn::abs::Neuron*, double>>& weights) override;

        std::vector<double> calculate() const override;

        EXCEPTION(IncompatibleVectorException);
    };

    template<typename NeuronType>
    requires std::is_base_of<nn::abs::BeginNeuron, NeuronType>::value
    class BeginLayer : public nn::abs::BeginLayer
    {
    protected:
        std::vector<std::shared_ptr<nn::abs::BeginNeuron>> neurons;

    public:
        BeginLayer(int numberOfNeurons);

        BeginLayer(int numberOfNeurons, const std::function<double(double)>& f);

        BeginLayer(std::vector<std::shared_ptr<nn::abs::BeginNeuron>> neurons, const std::function<double(double)>& f);

        BeginLayer(std::vector<std::shared_ptr<nn::abs::BeginNeuron>> neurons);

        BeginLayer() = default;

        void setNeuron(int index, std::shared_ptr<nn::abs::BeginNeuron> n);

        void setNeurons(std::vector<std::shared_ptr<nn::abs::BeginNeuron>> n);

        void setValues(const std::vector<double>& v) override;

        std::vector<double> calculate() const override;

        std::vector<std::shared_ptr<nn::abs::BeginNeuron>> getBeginNeurons() override;

        int getSize() const override;

        void connect(Layer* l) override;

        std::vector<std::shared_ptr<nn::abs::Neuron>> getNeurons() override;

        void setActivation(std::function<double(double)> f) override;

        void setBias(const std::vector<double>& bs) override;

        void setWeights(const std::vector<std::map<nn::abs::Neuron*, double>>& weights) override;

        EXCEPTION(IncompatibleVectorException);
    };
}

template<typename NeuronType>
requires std::is_base_of<nn::abs::Neuron, NeuronType>::value
nn::Layer<NeuronType>::Layer(int numberOfNeurons) : neurons(numberOfNeurons)
{
    for (auto& neuron : neurons)
        neuron = std::shared_ptr<nn::abs::Neuron>(dynamic_cast<nn::abs::Neuron*>(new NeuronType{}));
}

template<typename NeuronType>
requires std::is_base_of<nn::abs::Neuron, NeuronType>::value
int nn::Layer<NeuronType>::getSize() const
{
    return neurons.size();
}

template<typename NeuronType>
requires std::is_base_of<nn::abs::Neuron, NeuronType>::value
void nn::Layer<NeuronType>::connect(nn::abs::Layer* l)
{
    for (auto& neuronThisLayer : neurons)
        for (auto& neuronOtherLayer : l->getNeurons())
            neuronThisLayer->connect(neuronOtherLayer.get());
}

template<typename NeuronType>
requires std::is_base_of<nn::abs::Neuron, NeuronType>::value
std::vector<std::shared_ptr<nn::abs::Neuron>> nn::Layer<NeuronType>::getNeurons()
{
    return neurons;
}

template<typename NeuronType>
requires std::is_base_of<nn::abs::Neuron, NeuronType>::value
void nn::Layer<NeuronType>::setActivation(std::function<double(double)> f)
{
    for (auto& n : neurons)
        n->setActivation(f);
}

template<typename NeuronType>
requires std::is_base_of<nn::abs::Neuron, NeuronType>::value
nn::Layer<NeuronType>::Layer(std::vector<std::shared_ptr<nn::abs::Neuron>> neurons) : neurons(std::move(neurons)) {}

template<typename NeuronType>
requires std::is_base_of<nn::abs::Neuron, NeuronType>::value
void nn::Layer<NeuronType>::setBias(const std::vector<double>& bs)
{
    if (bs.size() != neurons.size())
        throw IncompatibleVectorException(
                "The size of the vector v must be equal to the size of the vector Layer<NeuronType>::neurons. "
                "v.size() = " + std::to_string(bs.size()) +
                " Layer<NeuronType>::neurons.size() = " + std::to_string(neurons.size()));

    for (int i = 0; i < neurons.size(); i++)
        neurons[i]->setB(bs[i]);
}

template<typename NeuronType>
requires std::is_base_of<nn::abs::Neuron, NeuronType>::value
void nn::Layer<NeuronType>::setWeights(const std::vector<std::map<nn::abs::Neuron*, double>>& weights)
{
    if (weights.size() != neurons.size())
        throw IncompatibleVectorException(
                "The size of the vector v must be equal to the size of the vector Layer<NeuronType>::neurons. "
                "weights.size() = " + std::to_string(weights.size()) +
                " Layer<NeuronType>::neurons.size() = " + std::to_string(neurons.size()));

    for (int i = 0; i < neurons.size(); i++)
        if (weights[i].size() != neurons[i]->getConnectionsNextLayer().size())
            throw IncompatibleVectorException(
                    "The size of the vector weights[i] must be equal to the size of the vector Layer<NeuronType>::neurons[]->connectionsNextLayer. "
                    "weights.size() = " + std::to_string(weights.size()) +
                    " Layer<NeuronType>::neurons.size() = " + std::to_string(neurons.size()) +
                    " i = " + std::to_string(i));

    for (int i = 0; i < neurons.size(); i++)
        neurons[i]->setWeights(weights[i]);
}

template<typename NeuronType>
requires std::is_base_of<nn::abs::Neuron, NeuronType>::value
nn::Layer<NeuronType>::Layer(int numberOfNeurons, const std::function<double(double)>& f) : neurons(numberOfNeurons)
{
        for (auto& neuron : neurons)
            neuron = std::shared_ptr<nn::abs::Neuron>(dynamic_cast<nn::abs::Neuron*>(new NeuronType{}));
        nn::Layer<NeuronType>::setActivation(f);
}

template<typename NeuronType>
requires std::is_base_of<nn::abs::Neuron, NeuronType>::value
nn::Layer<NeuronType>::Layer(std::vector<std::shared_ptr<nn::abs::Neuron>> neurons, std::function<double(double)> f) : neurons(std::move(neurons))
{
    nn::Layer<NeuronType>::setActivation(f);
}

template<typename NeuronType>
requires std::is_base_of<nn::abs::Neuron, NeuronType>::value
std::vector<double> nn::Layer<NeuronType>::calculate() const
{
    std::vector<double> retValue;
    retValue.reserve(neurons.size());
    for (auto& n : neurons)
        retValue.push_back(n->getValue());
    return retValue;
}

template<typename NeuronType>
requires std::is_base_of<nn::abs::BeginNeuron, NeuronType>::value
void nn::BeginLayer<NeuronType>::setValues(const std::vector<double>& v)
{
    if (v.size() != neurons.size())
        throw IncompatibleVectorException(
                "The size of the vector v must be equal to the size of the vector Layer<NeuronType>::neurons. "
                "v.size() = " + std::to_string(v.size()) +
                " Layer<NeuronType>::neurons.size() = " + std::to_string(neurons.size()));

    for (int i = 0; i < neurons.size(); i++)
        neurons[i]->setValue(v[i]);
}

template<typename NeuronType>
requires std::is_base_of<nn::abs::BeginNeuron, NeuronType>::value
nn::BeginLayer<NeuronType>::BeginLayer(int numberOfNeurons) : neurons(numberOfNeurons)
{
    for (auto& neuron : neurons)
        neuron = std::shared_ptr<nn::abs::BeginNeuron>(new NeuronType{});
}

template<typename NeuronType>
requires std::is_base_of<nn::abs::BeginNeuron, NeuronType>::value
std::vector<std::shared_ptr<nn::abs::BeginNeuron>> nn::BeginLayer<NeuronType>::getBeginNeurons()
{
    return neurons;
}

template<typename NeuronType>
requires std::is_base_of<nn::abs::BeginNeuron, NeuronType>::value
int nn::BeginLayer<NeuronType>::getSize() const
{
    return neurons.size();
}

template<typename NeuronType>
requires std::is_base_of<nn::abs::BeginNeuron, NeuronType>::value
void nn::BeginLayer<NeuronType>::connect(nn::abs::Layer* l)
{
    for (auto& neuronThisLayer : neurons)
        for (auto& neuronOtherLayer : l->getNeurons())
            neuronThisLayer->connect(neuronOtherLayer.get());
}

template<typename NeuronType>
requires std::is_base_of<nn::abs::BeginNeuron, NeuronType>::value
std::vector<std::shared_ptr<nn::abs::Neuron>> nn::BeginLayer<NeuronType>::getNeurons()
{
    std::vector<std::shared_ptr<nn::abs::Neuron>> retVec{neurons.size()};
    for (int i = 0; i < neurons.size(); i++)
        retVec[i] = std::shared_ptr<nn::abs::Neuron>{neurons[i]};
    return retVec;
}

template<typename NeuronType>
requires std::is_base_of<nn::abs::BeginNeuron, NeuronType>::value
void nn::BeginLayer<NeuronType>::setActivation(std::function<double(double)> f)
{
    for (auto& n : neurons)
        n->setActivation(f);
}

template<typename NeuronType>
requires std::is_base_of<nn::abs::BeginNeuron, NeuronType>::value
void nn::BeginLayer<NeuronType>::setNeuron(int index, std::shared_ptr<nn::abs::BeginNeuron> n)
{
    neurons[index] = std::move(n);
}

template<typename NeuronType>
requires std::is_base_of<nn::abs::BeginNeuron, NeuronType>::value
void nn::BeginLayer<NeuronType>::setNeurons(std::vector<std::shared_ptr<nn::abs::BeginNeuron>> n)
{
    neurons = std::move(n);
}

template<typename NeuronType>
requires std::is_base_of<nn::abs::BeginNeuron, NeuronType>::value
void nn::BeginLayer<NeuronType>::setBias(const std::vector<double>& bs)
{
    if (bs.size() != neurons.size())
        throw IncompatibleVectorException(
                "The size of the vector v must be equal to the size of the vector Layer<NeuronType>::neurons. "
                "v.size() = " + std::to_string(bs.size()) +
                " Layer<NeuronType>::neurons.size() = " + std::to_string(neurons.size()));

    for (int i = 0; i < neurons.size(); i++)
        neurons[i]->setB(bs[i]);
}

template<typename NeuronType>
requires std::is_base_of<nn::abs::BeginNeuron, NeuronType>::value
void nn::BeginLayer<NeuronType>::setWeights(const std::vector<std::map<nn::abs::Neuron*, double>>& weights)
{
    if (weights.size() != neurons.size())
        throw IncompatibleVectorException(
                "The size of the vector v must be equal to the size of the vector Layer<NeuronType>::neurons. "
                "weights.size() = " + std::to_string(weights.size()) +
                " Layer<NeuronType>::neurons.size() = " + std::to_string(neurons.size()));

    for (int i = 0; i < neurons.size(); i++)
        if (weights[i].size() != neurons[i]->getConnectionsNextLayer().size())
            throw IncompatibleVectorException(
                    "The size of the vector weights[i] must be equal to the size of the vector Layer<NeuronType>::neurons[]->connectionsNextLayer. "
                    "weights.size() = " + std::to_string(weights[i].size()) +
                    " Layer<NeuronType>::neurons.size() = " + std::to_string(neurons[i]->getConnectionsNextLayer().size()) +
                    " i = " + std::to_string(i));

    for (int i = 0; i < neurons.size(); i++)
        for (auto& w : weights[i])
            neurons[i]->setWeights(weights[i]);
}

template<typename NeuronType>
requires std::is_base_of<nn::abs::BeginNeuron, NeuronType>::value
nn::BeginLayer<NeuronType>::BeginLayer(int numberOfNeurons, const std::function<double(double)>& f) : neurons(numberOfNeurons)
{
    for (auto& neuron : neurons)
        neuron = std::shared_ptr<nn::abs::BeginNeuron>(dynamic_cast<nn::abs::BeginNeuron*>(new NeuronType{}));
    nn::BeginLayer<NeuronType>::setActivation(f);
}

template<typename NeuronType>
requires std::is_base_of<nn::abs::BeginNeuron, NeuronType>::value
nn::BeginLayer<NeuronType>::BeginLayer(std::vector<std::shared_ptr<nn::abs::BeginNeuron>> neurons,
                                       const std::function<double(double)>& f) : neurons(std::move(neurons))
{
    nn::BeginLayer<NeuronType>::setActivation(f);
}

template<typename NeuronType>
requires std::is_base_of<nn::abs::BeginNeuron, NeuronType>::value
nn::BeginLayer<NeuronType>::BeginLayer(std::vector<std::shared_ptr<nn::abs::BeginNeuron>> neurons) : neurons(std::move(neurons))
{

}

template<typename NeuronType>
requires std::is_base_of<nn::abs::BeginNeuron, NeuronType>::value
std::vector<double> nn::BeginLayer<NeuronType>::calculate() const
{
    std::vector<double> retValue;
    retValue.reserve(neurons.size());
    for (auto& n : neurons)
        retValue.push_back(n->getValue());
    return retValue;
}


#endif //NEURONAL_NETWORK_LAYER_H
