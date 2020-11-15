#ifndef NEURONAL_NETWORK_BACKPROPAGATOR_H
#define NEURONAL_NETWORK_BACKPROPAGATOR_H

#include "pch.h"

#include "abstract/Network.h"
#include "abstract/Backpropagator.h"

namespace nn
{
    class Backpropagator : public nn::abs::Backpropagator
    {
    private:
        struct NeuronGradient
        {
            std::shared_ptr<nn::abs::Neuron> n;
            double biasGradient;
            std::map<nn::abs::Neuron*, double> weightsGradient;
            double activationGradient;
        };

        void checkVectorSizes(const std::vector<std::vector<double>>& x,
                              const std::vector<std::vector<double>>& y) const;

        void errorCheck(const std::vector<std::vector<double>>& x, const std::vector<std::vector<double>>& y,
                        int epochs, int batchSize) const;

        void checkYVector(const std::vector<std::vector<double>>& y) const;

        void checkXVector(const std::vector<std::vector<double>>& x) const;

        void checkInitialized() const;

        void checkBatchSize(int batchSize, int numberOfSamples) const;

        void checkEpochs(int epochs) const;

        nn::abs::Network* net;
        std::shared_ptr<nn::abs::LossFunction> lossF;
        bool initialized = false;

    public:
        void initialize(nn::abs::Network* n, std::shared_ptr<nn::abs::LossFunction> lossF) override;

        void fit(const std::vector<std::vector<double>>& x,
                 const std::vector<std::vector<double>>& y, double learningRate, int epochs, int batchSize) override;

        void fit(const std::vector<std::vector<double>>& x,
                 const std::vector<std::vector<double>>& y, double learningRate, int epochs) override;

        EXCEPTION(InvalidVectorSize);

        EXCEPTION(InvalidBatchSize);

        EXCEPTION(InvalidEpochCount);

        EXCEPTION(NotInitializedException);
    };
}

#endif //NEURONAL_NETWORK_BACKPROPAGATOR_H
