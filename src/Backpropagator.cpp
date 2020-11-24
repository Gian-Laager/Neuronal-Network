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
    convertXYsToPairs(x, y, trainData);

    std::random_device rd;
    std::mt19937 g(rd());

    for (int i = 0; i < epochs; i++)
    {
        std::vector<std::vector<NeuronGradient>> gradient = initializeGradient();

        double loss = 0.0;
        calculateGradient(batchSize, trainData, gradient, loss);

        updateNetworkVariables(learningRate, batchSize, gradient);

        //TODO: Rewrite this as an if statement
#ifdef NN_NETWORK_PRINT_LOSS
        std::cout << "loss: " << loss / batchSize << "\n";
#endif
        std::shuffle(trainData.begin(), trainData.end(), g);
    }
}

void nn::Backpropagator::updateNetworkVariables(double learningRate, int batchSize,
                                                const std::vector<std::vector<NeuronGradient>>& gradient)
{
    for (int l = 0; l < this->net->getNumberOfLayers(); l++)
        for (int j = 0; j < this->net->getLayer(l)->getSize(); j++)
        {
            updateBiases(learningRate, batchSize, gradient, l, j);
            updateWeights(learningRate, batchSize, gradient, l, j);
        }
}

void nn::Backpropagator::updateWeights(double learningRate, int batchSize,
                                       const std::vector<std::vector<NeuronGradient>>& gradient, int l, int j)
{
    if (l > 0)
        for (auto& k : gradient[l][j].weightsGradient)
            this->net->getLayer(l)->getNeuron(j)->getConnectionPreviousLayer(k.first)->w +=
                    learningRate * k.second / batchSize;
}

void nn::Backpropagator::updateBiases(double learningRate, int batchSize,
                                      const std::vector<std::vector<NeuronGradient>>& gradient, int l, int j)
{
    this->net->getLayer(l)->getNeuron(j)->setB(
            this->net->getLayer(l)->getNeuron(j)->getB() +
            learningRate * gradient[l][j].biasGradient / batchSize);
}

void nn::Backpropagator::calculateGradient(int batchSize,
                                           const std::vector<std::pair<std::vector<double>, std::vector<double>>>& trainData,
                                           std::vector<std::vector<NeuronGradient>>& gradient, double& loss)
{
    for (int m = 0; m < batchSize; m++)
    {
        net->calculate(trainData[m].first);
#ifdef NN_NETWORK_PRINT_LOSS
        loss += getLoss(trainData, m);
#endif
        this->updateGradient(trainData, gradient, m);
    }
}

std::vector<std::vector<nn::Backpropagator::NeuronGradient>> nn::Backpropagator::initializeGradient()
{
    std::vector<std::vector<nn::Backpropagator::NeuronGradient>> gradient(this->net->getNumberOfLayers());
    for (int l = 0; l < this->net->getNumberOfLayers(); l++)
        gradient[l] = std::vector<nn::Backpropagator::NeuronGradient>(this->net->getLayer(l)->getSize());
    return gradient;
}

double
nn::Backpropagator::getLoss(const std::vector<std::pair<std::vector<double>, std::vector<double>>>& trainData,
                            int m)
{
    std::vector<double> predict = this->net->getLayer(net->getSize() -1)->calculate();
    double currentLoss = 0.0;
    for (int j = 0; j < predict.size(); j++)
        currentLoss += (*this->lossF)(predict[j], trainData[m].second[j]);
    return currentLoss / predict.size();
}

void
nn::Backpropagator::updateGradient(const std::vector<std::pair<std::vector<double>, std::vector<double>>>& trainData,
                                   std::vector<std::vector<NeuronGradient>>& gradient, int m)
{
    for (int l = this->net->getNumberOfLayers() - 1; l >= 0; l--)
    {
        gradient[l].reserve(this->net->getLayer(l)->getSize());
        for (int j = 0; j < this->net->getLayer(l)->getSize(); j++)
        {
            std::shared_ptr<nn::abs::Neuron> nlj = this->setUpGradient(l, j, gradient);
            this->updateActivationGradient(gradient, nlj, trainData, m, l, j);
            this->updateWeightsGradient(gradient, m, l, j, nlj);
            this->updateBiasGradient(m, l, j, nlj, gradient);
        }
    }
}

void nn::Backpropagator::updateBiasGradient(int m, int l, int j, const std::shared_ptr<nn::abs::Neuron>& nlj,
                                            std::vector<std::vector<NeuronGradient>>& gradient) const
{
    gradient[l][j].biasGradient += this->getBiasGradient(gradient, m, l, j, nlj);
}

void nn::Backpropagator::updateWeightsGradient(std::vector<std::vector<NeuronGradient>>& gradient, int batch, int l, int j,
                                               const std::shared_ptr<nn::abs::Neuron>& nlj) const
{
    if (l > 0)
        for (auto& nl_1k : this->net->getLayer(l - 1)->getNeurons())
            gradient[l][j].weightsGradient[nl_1k.get()] += this->getWeightGradient(gradient, batch, l, j, nlj, nl_1k);
}

double
nn::Backpropagator::getWeightGradient(const std::vector<std::vector<NeuronGradient>>& gradient, int m, int l, int j,
                                      const std::shared_ptr<nn::abs::Neuron>& nlj,
                                      const std::shared_ptr<nn::abs::Neuron>& nl_1k) const
{
    return nl_1k->getValue() * nlj->getActivation()->derivative(nlj->getZ()) *
           gradient[l][j].activationGradient / (m + 1);
}

double
nn::Backpropagator::getBiasGradient(const std::vector<std::vector<NeuronGradient>>& gradient, int m, int l, int j,
                                    const std::shared_ptr<nn::abs::Neuron>& nlj) const
{
    return nlj->getActivation()->derivative(nlj->getZ()) * gradient[l][j].activationGradient / (m + 1);
}

double
nn::Backpropagator::getActivationGradient(const std::vector<std::vector<NeuronGradient>>& gradient, int m, int l, int j,
                                          const std::shared_ptr<nn::abs::Neuron>& nlj) const
{
    double currentGradient = 0.0;
    for (int j_ = 0; j_ < this->net->getLayer(l + 1)->getSize(); j_++)
        currentGradient +=
                this->net->getLayer(l + 1)->getNeuron(j_)->getConnectionPreviousLayer(nlj.get())->w *
                this->net->getLayer(l + 1)->getNeuron(j_)->getActivation()->derivative(
                        this->net->getLayer(l + 1)->getNeuron(j_)->getZ()) *
                gradient[l + 1][j_].activationGradient / (m + 1);
    currentGradient /= this->net->getLayer(l + 1)->getSize();
    return currentGradient;
}

double nn::Backpropagator::getActivationGradientLastLayer(double yPred, double yTrue, int m,
                                                          int j) const { return this->lossF->derivative(yPred, yTrue); }

void nn::Backpropagator::convertXYsToPairs(const std::vector<std::vector<double>>& x,
                                           const std::vector<std::vector<double>>& y,
                                           std::vector<std::pair<std::vector<double>, std::vector<double>>>& trainData)
{
    for (int i = 0; i < x.size(); i++)
        trainData[i] = {x[i], y[i]};
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

void nn::Backpropagator::updateActivationGradient(std::vector<std::vector<NeuronGradient>>& gradient,
                                                  const std::shared_ptr<nn::abs::Neuron>& nlj,
                                                  const std::vector<std::pair<std::vector<double>, std::vector<double>>>& trainData,
                                                  int batch, int l, int j) const
{
    if (l == net->getNumberOfLayers() - 1)
        gradient[l][j].activationGradient += getActivationGradientLastLayer(nlj->getValue(), trainData[batch].second[j],
                                                                            batch, j);
    else
        gradient[l][j].activationGradient += getActivationGradient(gradient, batch, l, j, nlj);
}

std::shared_ptr<nn::abs::Neuron>
nn::Backpropagator::setUpGradient(int l, int j, std::vector<std::vector<NeuronGradient>>& gradient) const
{
    gradient[l][j].n = net->getLayer(l)->getNeuron(j);
    return gradient[l][j].n;
}
