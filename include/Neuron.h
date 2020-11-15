#ifndef NEURONAL_NETWORK_NEURON_H
#define NEURONAL_NETWORK_NEURON_H

#include "abstract/Neuron.h"

#include "Activations.h"

namespace nn
{
    struct Connection : public nn::abs::Connection
    {
        nn::abs::Neuron* from;
        nn::abs::Neuron* to;

        Connection(nn::abs::Neuron* from = nullptr, nn::abs::Neuron* to = nullptr);

        Connection() = default;
    };

    class Neuron : public nn::abs::Neuron
    {
    protected:
        double b = 0.0;

        std::map<nn::abs::Neuron*, std::shared_ptr<nn::abs::Connection>> connectionsNextLayer;
        std::map<nn::abs::Neuron*, std::shared_ptr<nn::abs::Connection>> connectionsPreviousLayer;
        std::shared_ptr<nn::abs::Activation> activationFunction = std::make_shared<nn::activations::Linear>();

        mutable bool cacheSet = false;
        mutable bool cacheZSet = false;
        mutable double cacheActivation;
        mutable double cacheZ;

        void isInValidKeyInWeights(const std::map<nn::abs::Neuron*, double>& weights) const;

    public:
        Neuron(std::map<nn::abs::Neuron*, std::shared_ptr<nn::abs::Connection>> connectionsNextLayer,
               std::map<nn::abs::Neuron*, std::shared_ptr<nn::abs::Connection>> connectionsPreviousLayer);

        Neuron() = default;

        void resetCache() const override;

        double getZ() const;

        std::map<nn::abs::Neuron*, std::shared_ptr<nn::abs::Connection>> getConnectionsNextLayer() override;

        void connect(nn::abs::Neuron* n) override;

        std::map<nn::abs::Neuron*, std::shared_ptr<nn::abs::Connection>> getConnectionsPreviousLayer() override;

        double getValue() const override;

        double getB() const override;

        void setB(double bias) override;

        void setActivation(std::shared_ptr<nn::abs::Activation> f) override;

        std::shared_ptr<const nn::abs::Activation> getActivation() const;

        void setWeights(std::map<nn::abs::Neuron*, double> weights) override;

        EXCEPTION(InvalidKeyInMapException);

        void appendToPreviousConnection(nn::abs::Neuron* n, std::shared_ptr<nn::abs::Connection> c) override;

    private:
        double multiplyPreviousLayersResultsByWeights() const;
    };

    class InputNeuron : public nn::abs::InputNeuron
    {
    protected:
        double b = 0.0;

        std::map<nn::abs::Neuron*, std::shared_ptr<nn::abs::Connection>> connectionsNextLayer;
        std::shared_ptr<nn::abs::Activation> activationFunction = std::make_shared<nn::activations::Linear>();

        mutable bool cacheSet = false;
        mutable bool cacheZSet = false;
        mutable double cacheActivation;
        mutable double cacheZ;

        void isInValidKeyInWeights(const std::map<nn::abs::Neuron*, double>& weights) const;

        double value = 0.0;
    public:
        InputNeuron(std::map<nn::abs::Neuron*, std::shared_ptr<nn::abs::Connection>> connectionsNextLayer);

        InputNeuron(double v, std::map<nn::abs::Neuron*, std::shared_ptr<nn::abs::Connection>> connectionsNextLayer);

        InputNeuron(double v);

        InputNeuron() = default;

        void resetCache() const override;

        std::map<nn::abs::Neuron*, std::shared_ptr<nn::abs::Connection>> getConnectionsNextLayer() override;

        void connect(nn::abs::Neuron* n) override;

        std::map<nn::abs::Neuron*, std::shared_ptr<nn::abs::Connection>> getConnectionsPreviousLayer() override;

        double getB() const override;

        void setB(double bias) override;

        void setActivation(std::shared_ptr<nn::abs::Activation> f) override;

        std::shared_ptr<const nn::abs::Activation> getActivation() const;

        void setWeights(std::map<nn::abs::Neuron*, double> weights) override;

        double getValue() const override;

        double getZ() const override;

        void setValue(double v) override;

        void appendToPreviousConnection(nn::abs::Neuron* n, std::shared_ptr<nn::abs::Connection> c) override;
    };
}

#endif //NEURONAL_NETWORK_NEURON_H
