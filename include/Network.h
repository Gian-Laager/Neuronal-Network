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
    public:
        Network(int initialNumberOfLayers);

        Network(int initialNumberOfLayers, const std::shared_ptr<nn::abs::BeginLayer>& firstLayer);

        Network(std::shared_ptr<nn::abs::BeginLayer> firstLayer);

        Network(const std::shared_ptr<nn::abs::BeginLayer>& firstLayer, std::vector<std::shared_ptr<nn::abs::Layer>> layers);

        Network() = default;

        int getNumberOfLayers() const override;

        int getCapacityOfLayers() const override;

        void pushLayer(std::shared_ptr<nn::abs::Layer> l) override;

        void setInputs(std::vector<double> values) override;

        std::shared_ptr<nn::abs::Layer> getLayer(int index) override;

        EXCEPTION(InvalidFirstLayerException);

        EXCEPTION(NoLayersGivenException);
    };
}

#endif //NEURONAL_NETWORK_NETWORK_H
