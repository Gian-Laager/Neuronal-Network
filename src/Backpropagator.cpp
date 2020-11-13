#include "Backpropagator.h"

void nn::Backpropagator::initialize(nn::abs::Network* n, std::shared_ptr<nn::abs::LossFunction> lossF)
{
    this->n = n;
    this->lossF = lossF;
    initialized = true;
}

void nn::Backpropagator::checkVectorSizes(const std::vector<std::vector<double>>& x,
                                          const std::vector<std::vector<double>>& y) const
{
    checkXVector(x);
    checkYVector(y);

    if (x.size() != y.size())
        throw InvalidVectorSize("The vectors x and y have to have the same size.");
}

void nn::Backpropagator::checkXVector(const std::vector<std::vector<double>>& x) const
{
    for (auto&& inVector : x)
        if (inVector.size() != this->n->getInputLayerSize())
            throw nn::Backpropagator::InvalidVectorSize(
                    "All vectors in the vector x have to have the same size as the networks first layers size.");
}

void nn::Backpropagator::checkYVector(const std::vector<std::vector<double>>& y) const
{
    for (auto&& outVector : y)
        if (outVector.size() != this->n->getOutputLayerSize())
            throw nn::Backpropagator::InvalidVectorSize(
                    "All vectors in the vector y have to have the same size as the networks last layers size.");
}

void nn::Backpropagator::fit(const std::vector<std::vector<double>>& x, const std::vector<std::vector<double>>& y,
                             int epochs, int batchSize)
{
    errorCheck(x, y, epochs, batchSize);
}

void nn::Backpropagator::errorCheck(const std::vector<std::vector<double>>& x,
                                    const std::vector<std::vector<double>>& y,
                                    int epochs, int batchSize) const
{
    checkInitialized();
    checkBatchSize(batchSize, x.size());
    checkEpochs(epochs);
    checkVectorSizes(x, y);
}

void nn::Backpropagator::checkEpochs(int epochs) const
{
    if (epochs < 0)
        throw nn::Backpropagator::InvalidEpochCount("Epochs must be grater than 0.");
}

void nn::Backpropagator::checkBatchSize(int batchSize, int numberOfSamples) const
{
    if (batchSize <= 0)
        throw nn::Backpropagator::InvalidBatchSize("Batch size must be grater than 0.");
    if (batchSize > numberOfSamples)
        throw nn::Backpropagator::InvalidBatchSize("Batch size must be smaller than the size of the x vector.");
}

void nn::Backpropagator::checkInitialized() const
{
    if (!this->initialized)
        throw nn::Backpropagator::NotInitializedException("Initialize must be called before fit.");
}

void nn::Backpropagator::fit(const std::vector<std::vector<double>>& x, const std::vector<std::vector<double>>& y,
                             int epochs)
{
    fit(x, y, epochs, x.size());
}
