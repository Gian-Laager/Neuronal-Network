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
            std::map<std::shared_ptr<nn::abs::Neuron>, double> weightsGradient;
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

        static void
        convertXYsToPairs(const std::vector<std::vector<double>>& x, const std::vector<std::vector<double>>& y,
                          std::vector<std::pair<std::vector<double>, std::vector<double>>>& trainData);

        double getActivationGradientLastLayer(double yPred, double yTrue, int m,
                                              int j) const;

        double getActivationGradient(int m, int l, int j,
                                     const std::shared_ptr<nn::abs::Neuron>& nlj) const;

        double getBiasGradient(int m, int l, int j,
                               const std::shared_ptr<nn::abs::Neuron>& nlj) const;

        double getWeightGradient(int m, int l, int j,
                                 const std::shared_ptr<nn::abs::Neuron>& nlj,
                                 double nlk1Activation) const;

        void updateActivationGradient(
                const std::shared_ptr<nn::abs::Neuron>& nlj,
                const std::vector<std::pair<std::vector<double>, std::vector<double>>>& trainData,
                int batch, int l, int j);

        void updateWeightsGradient(int batch, int l, int j,
                                   const std::shared_ptr<nn::abs::Neuron>& nlj);

        void updateBiasGradient(int m, int l, int j, const std::shared_ptr<nn::abs::Neuron>& nlj);

        void
        updateGradient(const std::vector<std::pair<std::vector<double>, std::vector<double>>>& trainData,
                       int m);

        double getLoss(const std::vector<std::pair<std::vector<double>, std::vector<double>>>& trainData,
                       int m);

        std::vector<std::vector<nn::Backpropagator::NeuronGradient>> initializeGradient();

        std::shared_ptr<nn::abs::Neuron> setUpGradient(int l, int j);

        void calculateGradient(int batchSize,
                               const std::vector<std::pair<std::vector<double>, std::vector<double>>>& trainData,
                               double& loss);

        void updateNetworkVariables(double learningRate, int batchSize);

        void updateBiases(double learningRate, int batchSize,
                          int l, int j);

        void updateWeights(double learningRate, int batchSize,
                           int l, int j);

        void resetGradient();

        nn::abs::Network* net;
        std::shared_ptr<nn::abs::LossFunction> lossF;
        bool initialized = false;
        std::vector<std::vector<NeuronGradient>> gradient;

    public:
        void initialize(nn::abs::Network* n, const std::shared_ptr<nn::abs::LossFunction>& lossF) override;

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
