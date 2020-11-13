#ifndef NEURONAL_NETWORK_LAYER_H
#define NEURONAL_NETWORK_LAYER_H

#define IS_VECTOR_COMPATIBLE(vector) if (vector.size() != neurons.size())\
        throw IncompatibleVectorException(\
                "The size of the vector " + std::string(#vector) + " must be equal to the size of the vector Layer<NeuronType>::neurons. "\
                + std::string(#vector) + " = " + std::to_string(vector.size()) +\
                " Layer<NeuronType>::neurons.size() = " + std::to_string(neurons.size()))

#define IS_VECTOR_MAP_COMPATIBLE(map, neurons) if (map.size() != neurons->getConnectionsNextLayer().size())\
            throw IncompatibleVectorException(\
                    "The size of the vector weights[i] must be equal to the size of the vector Layer<NeuronType>::neurons[]->connectionsNextLayer. "\
                    + std::string(#map) +".size() = " + std::to_string(map.size()) +\
                    " Layer<NeuronType>::"+ std::string(#neurons) +".size() = " + std::to_string(neurons->getConnectionsNextLayer().size()) +\
                    " i = " + std::to_string(i))

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

        Layer(int numberOfNeurons, std::shared_ptr<nn::abs::Activation> f);

        Layer(std::vector<std::shared_ptr<nn::abs::Neuron>> neurons, std::shared_ptr<nn::abs::Activation> f);

        Layer() = default;

        int getSize() const override;

        void connect(nn::abs::Layer* l) override;

        std::vector<std::shared_ptr<nn::abs::Neuron>> getNeurons() override;

        void setActivation(std::shared_ptr<nn::abs::Activation> f) override;

        void setBias(const std::vector<double>& bs) override;

        void setWeights(const std::vector<std::map<nn::abs::Neuron*, double>>& weights) override;

        std::vector<double> calculate() const override;

        std::vector<std::shared_ptr<const nn::abs::Neuron>> getNeurons() const override;

        void resetCaches() const override;

        EXCEPTION(IncompatibleVectorException);
    };

    template<typename NeuronType>
    requires std::is_base_of<nn::abs::InputNeuron, NeuronType>::value
    class InputLayer : public nn::abs::InputLayer
    {
    protected:
        std::vector<std::shared_ptr<nn::abs::InputNeuron>> neurons;

    public:
        void resetCaches() const override;

        InputLayer(int numberOfNeurons);

        InputLayer(int numberOfNeurons, std::shared_ptr<nn::abs::Activation> f);

        InputLayer(std::vector<std::shared_ptr<nn::abs::InputNeuron>> neurons,  std::shared_ptr<nn::abs::Activation> f);

        InputLayer(std::vector<std::shared_ptr<nn::abs::InputNeuron>> neurons);

        InputLayer() = default;

        void setNeuron(int index, std::shared_ptr<nn::abs::InputNeuron> n);

        void setNeurons(std::vector<std::shared_ptr<nn::abs::InputNeuron>> n);

        void setValues(const std::vector<double>& v) override;

        std::vector<double> calculate() const override;

        std::vector<std::shared_ptr<nn::abs::InputNeuron>> getInputNeurons() override;

        std::vector<std::shared_ptr<const nn::abs::Neuron>> getNeurons() const override;

        int getSize() const override;

        void connect(Layer* l) override;

        std::vector<std::shared_ptr<nn::abs::Neuron>> getNeurons() override;

        void setActivation(std::shared_ptr<nn::abs::Activation> f) override;

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
        neuron = std::shared_ptr<nn::abs::Neuron>(new NeuronType{});
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
void nn::Layer<NeuronType>::setActivation(std::shared_ptr<nn::abs::Activation> f)
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
        IS_VECTOR_COMPATIBLE(bs);

    for (int i = 0; i < neurons.size(); i++)
        neurons[i]->setB(bs[i]);
}

template<typename NeuronType>
requires std::is_base_of<nn::abs::Neuron, NeuronType>::value
void nn::Layer<NeuronType>::setWeights(const std::vector<std::map<nn::abs::Neuron*, double>>& weights)
{
    IS_VECTOR_COMPATIBLE(weights);

    for (int i = 0; i < neurons.size(); i++)
        IS_VECTOR_MAP_COMPATIBLE(weights[i], neurons[i]);

    for (int i = 0; i < neurons.size(); i++)
        neurons[i]->setWeights(weights[i]);
}

template<typename NeuronType>
requires std::is_base_of<nn::abs::Neuron, NeuronType>::value
nn::Layer<NeuronType>::Layer(int numberOfNeurons,  std::shared_ptr<nn::abs::Activation> f) : neurons(numberOfNeurons)
{
        for (auto& neuron : neurons)
            neuron = std::shared_ptr<nn::abs::Neuron>(new NeuronType{});
        nn::Layer<NeuronType>::setActivation(f);
}

template<typename NeuronType>
requires std::is_base_of<nn::abs::Neuron, NeuronType>::value
nn::Layer<NeuronType>::Layer(std::vector<std::shared_ptr<nn::abs::Neuron>> neurons, std::shared_ptr<nn::abs::Activation> f) : neurons(std::move(neurons))
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
requires std::is_base_of<nn::abs::Neuron, NeuronType>::value
void nn::Layer<NeuronType>::resetCaches() const
{
    for (auto& n : neurons)
        n->resetCache();
}

template<typename NeuronType>
requires std::is_base_of<nn::abs::Neuron, NeuronType>::value
std::vector<std::shared_ptr<const nn::abs::Neuron>> nn::Layer<NeuronType>::getNeurons() const
{
    return *(const std::vector<std::shared_ptr<const nn::abs::Neuron>>*) ((long) &neurons);
}

template<typename NeuronType>
requires std::is_base_of<nn::abs::InputNeuron, NeuronType>::value
void nn::InputLayer<NeuronType>::setValues(const std::vector<double>& v)
{
    IS_VECTOR_COMPATIBLE(v);
    for (int i = 0; i < neurons.size(); i++)
        neurons[i]->setValue(v[i]);
}

template<typename NeuronType>
requires std::is_base_of<nn::abs::InputNeuron, NeuronType>::value
nn::InputLayer<NeuronType>::InputLayer(int numberOfNeurons) : neurons(numberOfNeurons)
{
    for (auto& neuron : neurons)
        neuron = std::shared_ptr<nn::abs::InputNeuron>(new NeuronType{});
}

template<typename NeuronType>
requires std::is_base_of<nn::abs::InputNeuron, NeuronType>::value
std::vector<std::shared_ptr<nn::abs::InputNeuron>> nn::InputLayer<NeuronType>::getInputNeurons()
{
    return neurons;
}

template<typename NeuronType>
requires std::is_base_of<nn::abs::InputNeuron, NeuronType>::value
int nn::InputLayer<NeuronType>::getSize() const
{
    return neurons.size();
}

template<typename NeuronType>
requires std::is_base_of<nn::abs::InputNeuron, NeuronType>::value
void nn::InputLayer<NeuronType>::connect(nn::abs::Layer* l)
{
    for (auto& neuronThisLayer : neurons)
        for (auto& neuronOtherLayer : l->getNeurons())
            neuronThisLayer->connect(neuronOtherLayer.get());
}

template<typename NeuronType>
requires std::is_base_of<nn::abs::InputNeuron, NeuronType>::value
std::vector<std::shared_ptr<nn::abs::Neuron>> nn::InputLayer<NeuronType>::getNeurons()
{
    std::vector<std::shared_ptr<nn::abs::Neuron>> retVec{neurons.size()};
    for (int i = 0; i < neurons.size(); i++)
        retVec[i] = std::shared_ptr<nn::abs::Neuron>{neurons[i]};
    return retVec;
}

template<typename NeuronType>
requires std::is_base_of<nn::abs::InputNeuron, NeuronType>::value
void nn::InputLayer<NeuronType>::setActivation(std::shared_ptr<nn::abs::Activation> f)
{
    for (auto& n : neurons)
        n->setActivation(f);
}

template<typename NeuronType>
requires std::is_base_of<nn::abs::InputNeuron, NeuronType>::value
void nn::InputLayer<NeuronType>::setNeuron(int index, std::shared_ptr<nn::abs::InputNeuron> n)
{
    neurons[index] = std::move(n);
}

template<typename NeuronType>
requires std::is_base_of<nn::abs::InputNeuron, NeuronType>::value
void nn::InputLayer<NeuronType>::setNeurons(std::vector<std::shared_ptr<nn::abs::InputNeuron>> n)
{
    neurons = std::move(n);
}

template<typename NeuronType>
requires std::is_base_of<nn::abs::InputNeuron, NeuronType>::value
void nn::InputLayer<NeuronType>::setBias(const std::vector<double>& bs)
{
    IS_VECTOR_COMPATIBLE(bs);

    for (int i = 0; i < neurons.size(); i++)
        neurons[i]->setB(bs[i]);
}

template<typename NeuronType>
requires std::is_base_of<nn::abs::InputNeuron, NeuronType>::value
void nn::InputLayer<NeuronType>::setWeights(const std::vector<std::map<nn::abs::Neuron*, double>>& weights)
{
    IS_VECTOR_COMPATIBLE(weights);

    for (int i = 0; i < weights.size(); i++)
        IS_VECTOR_MAP_COMPATIBLE(weights[i], neurons[i]);


    for (int i = 0; i < neurons.size(); i++)
        for (auto& w : weights[i])
            neurons[i]->setWeights(weights[i]);
}

template<typename NeuronType>
requires std::is_base_of<nn::abs::InputNeuron, NeuronType>::value
nn::InputLayer<NeuronType>::InputLayer(int numberOfNeurons, std::shared_ptr<nn::abs::Activation> f) : neurons(numberOfNeurons)
{
    for (auto& neuron : neurons)
        neuron = std::shared_ptr<nn::abs::InputNeuron>(new NeuronType{});
    nn::InputLayer<NeuronType>::setActivation(f);
}

template<typename NeuronType>
requires std::is_base_of<nn::abs::InputNeuron, NeuronType>::value
nn::InputLayer<NeuronType>::InputLayer(std::vector<std::shared_ptr<nn::abs::InputNeuron>> neurons,
                                       std::shared_ptr<nn::abs::Activation> f) : neurons(std::move(neurons))
{
    nn::InputLayer<NeuronType>::setActivation(f);
}

template<typename NeuronType>
requires std::is_base_of<nn::abs::InputNeuron, NeuronType>::value
nn::InputLayer<NeuronType>::InputLayer(std::vector<std::shared_ptr<nn::abs::InputNeuron>> neurons) : neurons(std::move(neurons))
{

}

template<typename NeuronType>
requires std::is_base_of<nn::abs::InputNeuron, NeuronType>::value
std::vector<double> nn::InputLayer<NeuronType>::calculate() const
{
    std::vector<double> retValue;
    retValue.reserve(neurons.size());
    for (auto& n : neurons)
        retValue.push_back(n->getValue());
    return retValue;
}

template<typename NeuronType>
requires std::is_base_of<nn::abs::InputNeuron, NeuronType>::value
void nn::InputLayer<NeuronType>::resetCaches() const
{
    for (auto& n : neurons)
        n->resetCache();
}

template<typename NeuronType>
requires std::is_base_of<nn::abs::InputNeuron, NeuronType>::value
std::vector<std::shared_ptr<const nn::abs::Neuron>> nn::InputLayer<NeuronType>::getNeurons() const
{
    return *(const std::vector<std::shared_ptr<const nn::abs::Neuron>>*) ((long*) &neurons);
}


#endif //NEURONAL_NETWORK_LAYER_H
