#ifndef NEURONAL_NETWORK_NETWORK_H
#define NEURONAL_NETWORK_NETWORK_H

#include "abstract.h"

namespace nn
{
    class Network : public nn::abs::Network
    {
    protected:
        int size = 0;
        std::vector<std::shared_ptr<nn::abs::Layer>> layers;

        bool isBeginLayer(const std::shared_ptr<nn::abs::Layer>& l) const;

        void areLayersGiven() const;

    public:
        Network(int initialNumberOfLayers);

        Network(int initialNumberOfLayers, const std::shared_ptr<nn::abs::BeginLayer>& firstLayer);

        Network(std::shared_ptr<nn::abs::BeginLayer> firstLayer);

        Network(const std::shared_ptr<nn::abs::BeginLayer>& firstLayer,
                std::vector<std::shared_ptr<nn::abs::Layer>> layers);

        Network() = default;

        int getNumberOfLayers() const override;

        int getCapacityOfLayers() const override;

        void pushLayer(std::shared_ptr<nn::abs::Layer> l) override;

        void setInputs(std::vector<double> values) override;

        std::vector<double> calculate() const override;

        void setActivation(int index, std::function<double(double)> f) override;

        void setBias(int index, std::vector<double> bs) override;

        void setWeights(int index, const std::vector<std::map<nn::abs::Neuron*, double>>& weights) override;

        std::shared_ptr<nn::abs::Layer> getLayer(int index) override;

        int getSize() const override;

        void resetCaches() const override;

        EXCEPTION(InvalidFirstLayerException);

        EXCEPTION(NoLayersGivenException);
    };
}

#endif //NEURONAL_NETWORK_NETWORK_H
