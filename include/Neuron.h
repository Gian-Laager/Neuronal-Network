#ifndef NEURONAL_NETWORK_NEURON_H
#define NEURONAL_NETWORK_NEURON_H

#include "abstract/Neuron.h"

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
        std::function<double(double)> activationFunction = [](double z) -> double { return z; };

        mutable bool cacheSet = false;
        mutable double cache;

        void isInValidKeyInWeights(const std::map<nn::abs::Neuron*, double>& weights) const;

    public:
        Neuron(std::map<nn::abs::Neuron*, std::shared_ptr<nn::abs::Connection>> connectionsNextLayer,
               std::map<nn::abs::Neuron*, std::shared_ptr<nn::abs::Connection>> connectionsPreviousLayer);

        Neuron() = default;

        void resetCache() const override;

        std::map<nn::abs::Neuron*, std::shared_ptr<nn::abs::Connection>> getConnectionsNextLayer() override;

        void connect(nn::abs::Neuron* n) override;

        std::map<nn::abs::Neuron*, std::shared_ptr<nn::abs::Connection>> getConnectionsPreviousLayer() override;

        double getValue() const override;

        double getB() const override;

        void setB(double bias) override;

        void setActivation(std::function<double(double)> f) override;

        void setWeights(std::map<nn::abs::Neuron*, double> weights) override;

        EXCEPTION(InvalidKeyInMapException);

        void appendToPreviousConnection(nn::abs::Neuron* n, std::shared_ptr<nn::abs::Connection> c) override;

    private:
        double multiplyPreviousLayersResultsByWeights() const;
    };

    class BeginNeuron : public nn::abs::BeginNeuron
    {
    protected:
        double b = 0.0;

        std::map<nn::abs::Neuron*, std::shared_ptr<nn::abs::Connection>> connectionsNextLayer;
        std::function<double(double)> activationFunction = [](double z) -> double { return z; };

        mutable bool cacheSet = false;
        mutable double cache;

        void isInValidKeyInWeights(const std::map<nn::abs::Neuron*, double>& weights) const;

        double value = 0.0;
    public:
        BeginNeuron(std::map<nn::abs::Neuron*, std::shared_ptr<nn::abs::Connection>> connectionsNextLayer);

        BeginNeuron(double v, std::map<nn::abs::Neuron*, std::shared_ptr<nn::abs::Connection>> connectionsNextLayer);

        BeginNeuron(double v);

        BeginNeuron() = default;

        void resetCache() const override;

        std::map<nn::abs::Neuron*, std::shared_ptr<nn::abs::Connection>> getConnectionsNextLayer() override;

        void connect(nn::abs::Neuron* n) override;

        std::map<nn::abs::Neuron*, std::shared_ptr<nn::abs::Connection>> getConnectionsPreviousLayer() override;

        double getB() const override;

        void setB(double bias) override;

        void setActivation(std::function<double(double)> f) override;

        void setWeights(std::map<nn::abs::Neuron*, double> weights) override;

        double getValue() const override;

        void setValue(double v) override;

        void appendToPreviousConnection(nn::abs::Neuron* n, std::shared_ptr<nn::abs::Connection> c) override;
    };
}

#endif //NEURONAL_NETWORK_NEURON_H
