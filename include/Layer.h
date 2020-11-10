#ifndef NEURONAL_NETWORK_LAYER_H
#define NEURONAL_NETWORK_LAYER_H

#include "abstract/Layer.h"

namespace nn
{
    template<typename NeuronType>
    class Layer : public nn::abs::Layer
    {
    protected:
        std::vector<std::shared_ptr<nn::abs::Neuron>> neurons;

    public:
        Layer(int numberOfNeurons);

        int getSize() const override;

        void connect(nn::abs::Layer* l) override;

        std::vector<std::shared_ptr<nn::abs::Neuron>> getNeurons() override;
    };
}

template<typename NeuronType>
nn::Layer<NeuronType>::Layer(int numberOfNeurons) : neurons(numberOfNeurons)
{
    for (auto& neuron : neurons)
        neuron = std::make_shared<NeuronType>();
}

template<typename NeuronType>
int nn::Layer<NeuronType>::getSize() const
{
    return neurons.size();
}

template<typename NeuronType>
void nn::Layer<NeuronType>::connect(nn::abs::Layer* l)
{
    for (auto& neuronThisLayer : neurons)
        for (auto& neuronOtherLayer : l->getNeurons())
            neuronThisLayer->connect(neuronOtherLayer.get());
}

template<typename NeuronType>
std::vector<std::shared_ptr<nn::abs::Neuron>> nn::Layer<NeuronType>::getNeurons()
{
    return neurons;
}

#endif //NEURONAL_NETWORK_LAYER_H
