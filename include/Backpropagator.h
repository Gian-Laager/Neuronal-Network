#ifndef NEURONAL_NETWORK_BACKPROPAGATOR_H
#define NEURONAL_NETWORK_BACKPROPAGATOR_H

#include "abstract/Network.h"
#include "abstract/Backpropagator.h"

namespace nn
{
    class Backpropagator : public nn::abs::Backpropagator
    {
    private:
        static void CheckForErrors(const nn::abs::Network* n, const std::vector<std::vector<double>>& x,
                                                const std::vector<std::vector<double>>& y) ;
    public:
        void fit(nn::abs::Network* n, const std::vector<std::vector<double>>& x,
                 const std::vector<std::vector<double>>& y,
                 std::shared_ptr<nn::abs::LossFunction> lossF,
                 long batchSize, long epochs = 1) override;

        EXCEPTION(InvalidVectorSize);
    };
}

#endif //NEURONAL_NETWORK_BACKPROPAGATOR_H
