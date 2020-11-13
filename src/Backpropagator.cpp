#include "Backpropagator.h"

void nn::Backpropagator::fit(nn::abs::Network* n, const std::vector<std::vector<double>>& x,
                             const std::vector<std::vector<double>>& y, std::shared_ptr<nn::abs::LossFunction> lossF,
                             long batchSize, long epochs)
{
    CheckForErrors(n, x, y);


}

void nn::Backpropagator::CheckForErrors(const nn::abs::Network* n, const std::vector<std::vector<double>>& x,
                                    const std::vector<std::vector<double>>& y)
{
    for (auto&& inVector : x)
        if (inVector.size() != n->getInputLayerSize())
            throw nn::Backpropagator::InvalidVectorSize("All vectors in the vector x have to have the same size as the networks first layers size.");

    for (auto&& outVector : y)
        if (outVector.size() != n->getOutputLayerSize())
            throw nn::Backpropagator::InvalidVectorSize("All vectors in the vector y have to have the same size as the networks last layers size.");
}
