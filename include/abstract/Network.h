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

        virtual std::shared_ptr<nn::abs::Layer> getLayer(int index) = 0;
    };
}

#endif //NEURONAL_NETWORK_ABSTRACT_NETWORK_H
