#ifndef NEURONAL_NETWORK_ABSTRACT_LAYER_H
#define NEURONAL_NETWORK_ABSTRACT_LAYER_H

#include "abstract/Neuron.h"
#include "abstract/Activation.h"

namespace nn::abs
{
    class Layer
    {
    public:
        virtual int getSize() const = 0;

        virtual void connect(Layer* l) = 0;

        virtual const std::vector<double>& calculate() const = 0;

        virtual const std::vector<std::shared_ptr<nn::abs::Neuron>>& getNeurons() = 0;

        virtual std::vector<std::shared_ptr<const nn::abs::Neuron>> getNeurons() const = 0;

        virtual const std::shared_ptr<nn::abs::Neuron>& getNeuron(int index) = 0;

        virtual const std::shared_ptr<const nn::abs::Neuron>& getNeuron(int index) const = 0;

        virtual void setActivation(const std::shared_ptr<Activation>& f) = 0;

        virtual void setBias(const std::vector<double>& bs) = 0;

        virtual void setWeights(const std::vector<std::map<std::shared_ptr<nn::abs::Neuron>, double>>& weights) = 0;

        virtual void resetCaches() const = 0;
    };

    class InputLayer : public Layer
    {
    public:
        virtual void setValues(const std::vector<double>& v) = 0;

        virtual const std::vector<std::shared_ptr<InputNeuron>>& getInputNeurons() = 0;
    };
}

#endif //NEURONAL_NETWORK_ABSTRACT_LAYER_H
