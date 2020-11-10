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

    public:
        Neuron(std::map<nn::abs::Neuron*, std::shared_ptr<nn::abs::Connection>> connectionsNextLayer,
               std::map<nn::abs::Neuron*, std::shared_ptr<nn::abs::Connection>> connectionsPreviousLayer);

        Neuron() = default;

        std::map<nn::abs::Neuron*, std::shared_ptr<nn::abs::Connection>> getConnectionsNextLayer() override;

        void connect(nn::abs::Neuron* n) override;

        std::map<nn::abs::Neuron*, std::shared_ptr<nn::abs::Connection>> getConnectionsPreviousLayer() override;

        double getValue() const override;

        double getB() const override;

        void setB(double b) override;

        void setActivation(std::function<double(double)> f) override;
    };

class BeginNeuron : public nn::abs::BeginNeuron, public nn::Neuron
    {
        double value = 0.0;
    public:
        BeginNeuron(std::map<nn::abs::Neuron*, std::shared_ptr<nn::abs::Connection>> connectionsNextLayer);

        BeginNeuron(double v, std::map<nn::abs::Neuron*, std::shared_ptr<nn::abs::Connection>> connectionsNextLayer);

        BeginNeuron(double v);

        BeginNeuron() = default;

        std::map<nn::abs::Neuron*, std::shared_ptr<nn::abs::Connection>> getConnectionsNextLayer() override;

        void connect(nn::abs::Neuron* n) override;

        std::map<nn::abs::Neuron*, std::shared_ptr<nn::abs::Connection>> getConnectionsPreviousLayer() override;

        double getB() const override;

        void setB(double b) override;

        void setActivation(std::function<double(double)> f) override;

        double getValue() const override;

        void setValue(double v) override;
    };
}

#endif //NEURONAL_NETWORK_NEURON_H
