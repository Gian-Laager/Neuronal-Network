#ifndef NEURONAL_NETWORK_SYCL_LAYER_H
#define NEURONAL_NETWORK_SYCL_LAYER_H

#include "abstract/Layer.h"

namespace nn::sycl
{
    class Layer : public nn::abs::Layer
    {
    public:
        int getSize() const override;

        void connect(const std::shared_ptr<nn::abs::Layer>& l) override;

        std::vector<double> calculate() const override;

        const std::vector<std::shared_ptr<nn::abs::Neuron>>& getNeurons() override;

        std::vector<std::shared_ptr<const nn::abs::Neuron>> getNeurons() const override;

        const std::shared_ptr<nn::abs::Neuron>& getNeuron(int index) override;

        const std::shared_ptr<const nn::abs::Neuron>& getNeuron(int index) const override;

        void setActivation(const std::shared_ptr<nn::abs::Activation>& f) override;

        void setBias(const std::vector<double>& bs) override;

        void setWeights(const std::vector<std::map<nn::abs::Neuron*, double>>& weights) override;

        void resetCaches() const override;
    };
}

#endif //NEURONAL_NETWORK_SYCL_LAYER_H
