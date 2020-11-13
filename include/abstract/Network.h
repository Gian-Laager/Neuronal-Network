#ifndef NEURONAL_NETWORK_ABSTRACT_NETWORK_H
#define NEURONAL_NETWORK_ABSTRACT_NETWORK_H

#include "abstract/Backpropagator.h"
#include "abstract/Activation.h"
#include "abstract/Layer.h"

namespace nn::abs
{
    class Network
    {
    public:
        virtual int getNumberOfLayers() const = 0;

        virtual int getCapacityOfLayers() const = 0;

        virtual void pushLayer(std::shared_ptr<nn::abs::Layer> l) = 0;

        virtual void setInputs(std::vector<double> values) = 0;

        virtual std::vector<double> calculate() const = 0;

        virtual std::shared_ptr<const nn::abs::Layer> getLayer(int index) const = 0;

        virtual std::shared_ptr<nn::abs::Layer> getLayer(int index) = 0;

        virtual int getSize() const = 0;

        virtual void setActivation(int index, std::shared_ptr<nn::abs::Activation> f) = 0;

        virtual void setBias(int index, std::vector<double> bs) = 0;

        virtual void setWeights(int index, const std::vector<std::map<nn::abs::Neuron*, double>>& weights) = 0;

        virtual void resetCaches() const = 0;

        virtual void setBackpropagator(std::shared_ptr<nn::abs::Backpropagator> backprop) = 0;

        virtual void initializeFitting(std::shared_ptr<nn::abs::LossFunction> lossF) = 0;

        virtual void fit(const std::vector<std::vector<double>>& x,
                         const std::vector<std::vector<double>>& y, int epochs, int batchSize) = 0;

        virtual void fit(const std::vector<std::vector<double>>& x,
                         const std::vector<std::vector<double>>& y, int epochs) = 0;

        virtual int getInputLayerSize() const = 0;

        virtual int getOutputLayerSize() const = 0;
    };
}

#endif //NEURONAL_NETWORK_ABSTRACT_NETWORK_H
