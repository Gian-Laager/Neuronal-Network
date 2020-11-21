#include "Backpropagator.h"

void nn::Backpropagator::initialize(nn::abs::Network* n, std::shared_ptr<nn::abs::LossFunction> lossF)
{
    this->net = n;
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
        if (inVector.size() != this->net->getInputLayerSize())
            throw nn::Backpropagator::InvalidVectorSize(
                    "All vectors in the vector x have to have the same size as the networks first layers size.");
}

void nn::Backpropagator::checkYVector(const std::vector<std::vector<double>>& y) const
{
    for (auto&& outVector : y)
        if (outVector.size() != this->net->getOutputLayerSize())
            throw nn::Backpropagator::InvalidVectorSize(
                    "All vectors in the vector y have to have the same size as the networks last layers size.");
}

void nn::Backpropagator::fit(const std::vector<std::vector<double>>& x, const std::vector<std::vector<double>>& y,
                             double learningRate, int epochs, int batchSize)
{
    errorCheck(x, y, epochs, batchSize);

    std::vector<std::pair<std::vector<double>, std::vector<double>>> trainData(x.size());
    for (int i = 0; i < x.size(); i++)
        trainData[i] = {x[i], y[i]};

    std::random_device rd;
    std::mt19937 g(rd());

    for (int i = 0; i < epochs; i++)
    {
        std::vector<std::vector<NeuronGradient>> gradient(net->getNumberOfLayers());
        for (int l = 0; l < net->getNumberOfLayers(); l++)
            gradient[l] = std::vector<NeuronGradient>(net->getLayer(l)->getSize());

        for (int m = 0; m < batchSize; m++)
        {
            std::vector<double> predict = net->calculate(trainData[m].first);

            for (int l = net->getNumberOfLayers() - 1; l >= 0; l--)
            {
                gradient[l].reserve(net->getLayer(l)->getSize());
                for (int j = 0; j < net->getLayer(l)->getSize(); j++)
                {
                    std::shared_ptr<nn::abs::Neuron> nlj = net->getLayer(l)->getNeurons()[j];
                    gradient[l][j].n = nlj;

                    if (l == net->getNumberOfLayers() - 1)
                        gradient[l][j].activationGradient += lossF->derivative(nlj->getValue(), trainData[m].second[j]);
                    else
                    {
                        for (int j_ = 0; j_ < net->getLayer(l + 1)->getSize(); j_++)
                            gradient[l][j].activationGradient +=
                                    net->getLayer(l + 1)->getNeurons()[j_]->getConnectionsPreviousLayer()[nlj.get()]->w *
                                    net->getLayer(l + 1)->getNeurons()[j_]->getActivation()->derivative(
                                            net->getLayer(l + 1)->getNeurons()[j_]->getZ()) *
                                    gradient[l + 1][j_].activationGradient / (m + 1);
                        gradient[l][j].activationGradient /= net->getLayer(l + 1)->getSize();
                    }

                    if (l > 0)
                        for (auto& nl_1k : net->getLayer(l - 1)->getNeurons())
                            gradient[l][j].weightsGradient[nl_1k.get()] +=
                                    nl_1k->getValue() * nlj->getActivation()->derivative(nlj->getZ()) *
                                    gradient[l][j].activationGradient / (m + 1);

                    gradient[l][j].biasGradient +=
                            nlj->getActivation()->derivative(nlj->getZ()) * gradient[l][j].activationGradient / (m + 1);

                }
            }
        }

        for (int l = 0; l < net->getNumberOfLayers(); l++)
            for (int j = 0; j < net->getLayer(l)->getSize(); j++)
            {
                net->getLayer(l)->getNeurons()[j]->setB(
                        net->getLayer(l)->getNeurons()[j]->getB() + learningRate *  gradient[l][j].biasGradient / batchSize);
                if (l > 0)
                    for (auto& k : gradient[l][j].weightsGradient)
                        net->getLayer(l)->getNeurons()[j]->getConnectionsPreviousLayer()[k.first]->w += learningRate * k.second / batchSize;
            }
        std::shuffle(trainData.begin(), trainData.end(), g);
    }
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
                             double learningRate,
                             int epochs)
{
    fit(x, y, learningRate, epochs, x.size());
}
