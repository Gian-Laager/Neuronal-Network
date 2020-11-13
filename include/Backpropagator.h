#ifndef NEURONAL_NETWORK_BACKPROPAGATOR_H
#define NEURONAL_NETWORK_BACKPROPAGATOR_H

#include "abstract/Network.h"
#include "abstract/Backpropagator.h"

namespace nn
{
    class Backpropagator : public nn::abs::Backpropagator
    {
    private:
        void checkVectorSizes(const std::vector<std::vector<double>>& x,
                              const std::vector<std::vector<double>>& y) const;

        void errorCheck(const std::vector<std::vector<double>>& x, const std::vector<std::vector<double>>& y,
                        int epochs, int batchSize) const;

        void checkYVector(const std::vector<std::vector<double>>& y) const;

        void checkXVector(const std::vector<std::vector<double>>& x) const;

        void checkInitialized() const;

        void checkBatchSize(int batchSize, int numberOfSamples) const;

        void checkEpochs(int epochs) const;

        nn::abs::Network* n;
        std::shared_ptr<nn::abs::LossFunction> lossF;
        bool initialized = false;

    public:
        void initialize(nn::abs::Network* n, std::shared_ptr<nn::abs::LossFunction> lossF) override;

        void fit(const std::vector<std::vector<double>>& x,
                 const std::vector<std::vector<double>>& y, int epochs, int batchSize) override;

        void fit(const std::vector<std::vector<double>>& x,
                 const std::vector<std::vector<double>>& y, int epochs) override;

        EXCEPTION(InvalidVectorSize);

        EXCEPTION(InvalidBatchSize);

        EXCEPTION(InvalidEpochCount);

        EXCEPTION(NotInitializedException);
    };
}

#endif //NEURONAL_NETWORK_BACKPROPAGATOR_H
