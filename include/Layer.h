#ifndef NEURONAL_NETWORK_LAYER_H
#define NEURONAL_NETWORK_LAYER_H

#include "abstract/Layer.h"

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

        int getSize() const override;

        void connect(nn::abs::Layer* l) override;

        std::vector<std::shared_ptr<nn::abs::Neuron>> getNeurons() override;

        void setActivation(std::function<double(double)> f) override;
    };

    template<typename NeuronType>
    requires std::is_base_of<nn::abs::BeginNeuron, NeuronType>::value
    class BeginLayer : public nn::abs::BeginLayer
    {
    protected:
        std::vector<std::shared_ptr<nn::abs::BeginNeuron>> neurons;

    public:
        BeginLayer(int numberOfNeurons);

        void setValues(const std::vector<double>& v) override;

        std::vector<std::shared_ptr<nn::abs::BeginNeuron>> getBeginNeurons() override;

        int getSize() const override;

        void connect(Layer* l) override;

        std::vector<std::shared_ptr<nn::abs::Neuron>> getNeurons() override;

        void setActivation(std::function<double(double)> f) override;

        class IncompatibleVectorException : public std::exception
        {
        public:
            std::string message;

            const char* what() const noexcept override;

            IncompatibleVectorException(std::string msg);
        };
    };
}

template<typename NeuronType>
requires std::is_base_of<nn::abs::Neuron, NeuronType>::value
nn::Layer<NeuronType>::Layer(int numberOfNeurons) : neurons(numberOfNeurons)
{
    for (auto& neuron : neurons)
        neuron = std::shared_ptr<nn::abs::Neuron>((nn::abs::Neuron*) ((long) new NeuronType{}));
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

}

template<typename NeuronType>
requires std::is_base_of<nn::abs::BeginNeuron, NeuronType>::value
nn::BeginLayer<NeuronType>::IncompatibleVectorException::IncompatibleVectorException(std::string msg) : message(
        std::move(msg)) {}

template<typename NeuronType>
requires std::is_base_of<nn::abs::BeginNeuron, NeuronType>::value
const char* nn::BeginLayer<NeuronType>::IncompatibleVectorException::what() const noexcept
{
    return message.c_str();
}

#endif //NEURONAL_NETWORK_LAYER_H
