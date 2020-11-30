#ifndef NEURONAL_NETWORK_NETWORK_H
#define NEURONAL_NETWORK_NETWORK_H

#include "abstract.h"
#include "Backpropagator.h"

namespace nn
{
    class Network : public nn::abs::Network
    {
    protected:
        int size = 0;

        bool fittingInitialized = false;

        std::vector<std::shared_ptr<nn::abs::Layer>> layers;

        mutable std::shared_ptr<nn::abs::Backpropagator> backpropagator;

        bool isInputLayer(const std::shared_ptr<nn::abs::Layer>& l) const;

        void areLayersGiven() const;

    public:
        Network(int initialNumberOfLayers);

        Network(int initialNumberOfLayers, const std::shared_ptr<nn::abs::InputLayer>& firstLayer);

        Network(const std::shared_ptr<nn::abs::InputLayer>& firstLayer);

        Network(const std::shared_ptr<nn::abs::InputLayer>& firstLayer,
                const std::vector<std::shared_ptr<nn::abs::Layer>>& layers);

        Network();

        int getNumberOfLayers() const override;

        int getCapacityOfLayers() const override;

        void pushLayer(const std::shared_ptr<nn::abs::Layer>& l) override;

        void setInputs(const std::vector<double>& values) override;

        std::vector<double> calculate() override;

        std::vector<double> calculate(std::vector<double> values) override;

        void setActivation(int index, const std::shared_ptr<nn::abs::Activation>& f) override;

        void setBias(int index, std::vector<double> bs) override;

        void setWeights(int index, const std::vector<std::map<abs::Neuron*, double>>& weights) override;

        std::shared_ptr<const nn::abs::Layer> getLayer(int index) const override;

        const std::shared_ptr<nn::abs::Layer>& getLayer(int index) override;

        int getSize() const override;

        void resetCaches() const override;

        void setBackpropagator(const std::shared_ptr<nn::abs::Backpropagator>& b) override;

        void initializeFitting(const std::shared_ptr<nn::abs::LossFunction>& lossF) override;

        void fit(const std::vector<std::vector<double>>& x,
                 const std::vector<std::vector<double>>& y, double learningRate, int epochs, int batchSize) override;

        void fit(const std::vector<std::vector<double>>& x,
                 const std::vector<std::vector<double>>& y, double learningRate, int epochs) override;

        int getInputLayerSize() const override;

        int getOutputLayerSize() const override;

        EXCEPTION(InvalidFirstLayerException);

        EXCEPTION(NoLayersGivenException);

        EXCEPTION(FitNotInitializedExeption);
    };
}

#endif //NEURONAL_NETWORK_NETWORK_H
