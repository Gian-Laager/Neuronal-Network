#ifndef NEURONAL_NETWORK_ABSTRACT_NEURON_H
#define NEURONAL_NETWORK_ABSTRACT_NEURON_H

#include "pch.h"

namespace nn::abs
{
    struct Connection
    {
        double w = 0.0;
    };

    class Neuron
    {
    public:
        virtual std::map<Neuron*, std::shared_ptr<nn::abs::Connection>> getConnectionsNextLayer() = 0;

        virtual std::map<Neuron*, std::shared_ptr<nn::abs::Connection>> getConnectionsPreviousLayer() = 0;

        virtual void connect(Neuron* n) = 0;

        virtual double getValue() const = 0;

        virtual double getB() const = 0;

        virtual void setB(double bias) = 0;

        virtual void setActivation(std::function<double(double)> f) = 0;

        virtual void setWeights(std::map<Neuron*, double> weights) = 0;
    };

    class BeginNeuron : public virtual nn::abs::Neuron
    {
    public:
        virtual void setValue(double v) = 0;
    };
}

#endif //NEURONAL_NETWORK_ABSTRACT_NEURON_H
