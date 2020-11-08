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
    private:
        std::vector<std::shared_ptr<nn::abs::Connection>> connectionsNextLayer;
        std::vector<std::shared_ptr<nn::abs::Connection>> connectionsPreviousLayer;

    public:
        Neuron(std::vector<nn::Connection*> connectionsNextLayer,
               std::vector<nn::Connection*> connectionsPreviousLayer);

        Neuron() = default;

        std::vector<std::shared_ptr<nn::abs::Connection>> getConnectionsNextLayer() override;
        std::vector<std::shared_ptr<nn::abs::Connection>> getConnectionsPreviousLayer() override;

        void connect(nn::abs::Neuron* n) override;
    };
}

#endif //NEURONAL_NETWORK_NEURON_H
