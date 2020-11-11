#ifndef NEURONAL_NETWORK_ABSTRACT_NETWORK_H
#define NEURONAL_NETWORK_ABSTRACT_NETWORK_H

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

        virtual std::shared_ptr<nn::abs::Layer> getLayer(int index) = 0;

        virtual int getSize() const = 0;

        virtual void setActivation(int index, std::function<double(double)> f) = 0;

        virtual void setBias(int index, std::vector<double> bs) = 0;

        virtual void setWeights(int index, const std::vector<std::map<nn::abs::Neuron*, double>>& weights) = 0;
    };
}

#endif //NEURONAL_NETWORK_ABSTRACT_NETWORK_H
