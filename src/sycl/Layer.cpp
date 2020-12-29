#include "sycl/Layer.h"

int nn::sycl::Layer::getSize() const
{
    return size;
}

void nn::sycl::Layer::connect(const std::shared_ptr<nn::abs::Layer>& l)
{
    for (auto& neuronThisLayer : getNeurons())
        for (auto& neuronsNextLayer : l->getNeurons())
            neuronThisLayer->connect(neuronsNextLayer.get());
}

std::vector<double> nn::sycl::Layer::calculate() const
{
    if (!connectionPreviousLayer)
        return std::move(calculateWithNonSyclNeurons());
    return std::move(calculateSyclWithVectorReturn());
}

const std::vector<std::shared_ptr<nn::abs::Neuron>>& nn::sycl::Layer::getNeurons()
{
    if (!neuronsSet)
        setNeurons();
    return neurons;
}

std::vector<std::shared_ptr<const nn::abs::Neuron>> nn::sycl::Layer::getNeurons() const
{
    if (!neuronsSet)
        setNeurons();
    return *(std::vector<std::shared_ptr<const nn::abs::Neuron>>*) (long) &neurons;
}

const std::shared_ptr<nn::abs::Neuron>& nn::sycl::Layer::getNeuron(int index)
{
    if (!neuronsSet)
        setNeurons();
    return neurons[index];
}

const std::shared_ptr<const nn::abs::Neuron>& nn::sycl::Layer::getNeuron(int index) const
{
    if (!neuronsSet)
        setNeurons();
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wreturn-stack-address"
    return (std::shared_ptr<const nn::abs::Neuron>) neurons[index];
#pragma clang diagnostic pop
}

void nn::sycl::Layer::setActivation(const std::shared_ptr<nn::abs::Activation>& f)
{
    activationFunction = f;
}

void nn::sycl::Layer::setBias(const std::vector<double>& bs)
{
    if (bs.size() != size)
        throw IncompatibleVectorException("The vector of biases has to have the same size as the layer.");
    auto biases_temp = cl::sycl::buffer<double, 1>{bs.data(), cl::sycl::range<1>{bs.size()}};
    biases = cl::sycl::buffer<double, 1>{cl::sycl::range<1>{bs.size()}};
    queue.submit([&](cl::sycl::handler& cgh) {
        auto biases_acc = biases.get_access<cl::sycl::access::mode::write>();
        auto temp_acc = biases_temp.get_access<cl::sycl::access::mode::read>();
        cgh.parallel_for(biases_acc.get_range(), [=](cl::sycl::id<1> i) {
            biases_acc[i] = temp_acc[i];
        });
    });
    queue.wait();
}

void nn::sycl::Layer::setWeights(const std::vector<std::map<nn::abs::Neuron*, double>>& w)
{

}

void nn::sycl::Layer::resetCaches() const
{
    valuesSet = false;
}

nn::sycl::Layer::Layer(int numberOfNeurons) : size(numberOfNeurons),
                                              biases(cl::sycl::range<1>{(unsigned long) size}),
                                              values(cl::sycl::range<1>{(unsigned long) size}),
                                              queue(cl::sycl::default_selector{}
                                              )
{
    initValues();
}

void nn::sycl::Layer::setNeurons() const
{
    setUpNeuronsVector();
    for (int i = 0; i < size; i++)
        setNeuronsIndex(i);

    neuronsSet = true;
    for (int i = 0; i < size; i++)
        setNeuronsConnectionIndex(i);
}

void nn::sycl::Layer::setUpNeuronsVector() const
{
    neurons = std::vector<std::shared_ptr<nn::abs::Neuron>>();
    neurons.reserve(size);
}

void nn::sycl::Layer::setNeuronsIndex(int i) const
{
    neurons.emplace_back(std::make_shared<nn::Neuron>());
    setNeuronsBiasIndex(i);
    neurons[i]->setActivation(activationFunction);
}

void nn::sycl::Layer::setNeuronsBiasIndex(int i) const
{
    auto& b = const_cast<trisycl::buffer<double, 1>&>(biases);
    queue.submit([&](trisycl::handler& cgh) {
        neurons[i]->setB(b.get_access<trisycl::access::mode::read>()[i]);
    });
    queue.wait();
}

nn::sycl::Layer::Layer(int numberOfNeurons, const std::shared_ptr<nn::abs::Activation>& f) : size(numberOfNeurons),
                                                                                             biases(cl::sycl::range<1>{
                                                                                                     (unsigned long) size}),
                                                                                             values(cl::sycl::range<1>{
                                                                                                     (unsigned long) size}),
                                                                                             queue(cl::sycl::default_selector{}
                                                                                             )
{
    initValues();
    setActivation(f);
}

void
nn::sycl::Layer::setConnectionPreviousLayer(const std::shared_ptr<nn::sycl::abs::Connection>& connectionPreviousLayer)
{
    this->connectionPreviousLayer = connectionPreviousLayer;
}

void nn::sycl::Layer::setConnectionNextLayer(const std::shared_ptr<nn::sycl::abs::Connection>& connectionNextLayer)
{
    this->connectionNextLayer = connectionNextLayer;
}

void nn::sycl::Layer::connect(const std::shared_ptr<nn::sycl::abs::Layer>& l)
{
    std::shared_ptr<nn::sycl::abs::Connection> c = std::make_shared<nn::sycl::abs::Connection>(this, l.get());
    setConnectionNextLayer(c);
    l->setConnectionPreviousLayer(c);
}

void nn::sycl::Layer::setNeuronsConnectionIndex(int i) const
{
    if (connectionNextLayer != nullptr)
        for (int j = 0; j < connectionNextLayer->to->getSize(); j++)
            neurons[i]->connect(connectionNextLayer->to->getNeuron(j).get());

    if (connectionPreviousLayer != nullptr)
        for (int j = 0; j < connectionPreviousLayer->from->getSize(); j++)
            connectionPreviousLayer->from->getNeuron(j)->connect(neurons[i].get());
}

void nn::sycl::Layer::setWeights(const cl::sycl::buffer<nn::abs::Connection, 2>& w)
{
    checkForErrorsSetWeightsBuffer(w);
    connectionNextLayer->weights = w;
}

void nn::sycl::Layer::checkForErrorsSetWeightsBuffer(const cl::sycl::buffer<nn::abs::Connection, 2>& w) const
{
    if (!this->connectionNextLayer)
        throw nn::sycl::Layer::NoConnectionException(
                "Connection to a next layer must be specified to execute this function");
    if (w.get_range()[1] != this->getSize() || w.get_range()[0] != this->connectionNextLayer->to->getSize())
        throw nn::sycl::Layer::IncompatibleVectorException(
                "Given buffer must have same size in first dimension as the next layers size and the second dimension must be equal to the size of the current layer");
}

std::vector<double> nn::sycl::Layer::calculateWithNonSyclNeurons() const
{
    std::vector<double> valuesVec(getSize());
    for (auto& n : getNeurons())
        valuesVec.emplace_back(n->getValue());
    return std::move(valuesVec);
}

std::vector<double> nn::sycl::Layer::calculateSyclWithVectorReturn() const
{
    valuesSet = true;
    auto values_acc = calculateSycl().get_access<cl::sycl::access::mode::read>();
    return std::vector<double>(values_acc.begin(), values_acc.end());
}

cl::sycl::buffer<double, 1>
nn::sycl::Layer::calculateSycl() const
{
    auto input = connectionPreviousLayer->from->calculateSycl();
//    queue.submit([&](cl::sycl::handler& cgh) {
//        auto values_acc = values.get_access<cl::sycl::access::mode::write>();
//        auto bias_acc = const_cast<cl::sycl::buffer<double, 1>&>(biases).get_access<cl::sycl::access::mode::read>();
//        auto weights_acc = connectionPreviousLayer->weights.get_access<cl::sycl::access::mode::read>();
//        auto activation = cl::sycl::buffer<nn::abs::Activation, 1>{activationFunction.get(), cl::sycl::range<1>{
//                1}}.get_access<cl::sycl::access::mode::read>();
//        auto input_acc = input.get_access<cl::sycl::access::mode::read>();
//        cgh.parallel_for(
//                cl::sycl::range<1>{(unsigned long) getSize()},
//                [=](cl::sycl::id<1> i) {
//                    for (unsigned long j = 0; j < connectionPreviousLayer->from->getSize(); j++)
//                    {
//                        values_acc[i] += input_acc[j] * weights_acc[cl::sycl::id<2>{i[0], j}].w;
//                        std::cout << "Weights at index " << i[0] << ", " << j << ": "
//                                  << weights_acc[cl::sycl::id<2>{i[0], j}].w << std::endl;
//
//                        if (i[0] == getSize() - 1)
//                            std::cout << "Input at index " << j << ": " << input_acc[j] + bias_acc[i] << std::endl;
//                    }
//                    std::cout << "Z " << values_acc[i[0]] << " at index " << i[0] << std::endl;
//
//                });
//    });
//    queue.wait();
    queue.submit([&](cl::sycl::handler& cgh) {
        auto values_acc = values.get_access<cl::sycl::access::mode::write>();
        auto bias_acc = const_cast<cl::sycl::buffer<double, 1>&>(biases).get_access<cl::sycl::access::mode::read>();
        auto weights_acc = connectionPreviousLayer->weights.get_access<cl::sycl::access::mode::read>();
        auto activation = cl::sycl::buffer<nn::abs::Activation, 1>{activationFunction.get(), cl::sycl::range<1>{
                1}}.get_access<cl::sycl::access::mode::read>();
        auto input_acc = input.get_access<cl::sycl::access::mode::read>();

        cgh.parallel_for(cl::sycl::range<1>{(unsigned long) getSize()}, [=](cl::sycl::id<1> i) {
            for (unsigned long j = 0; j < connectionPreviousLayer->from->getSize(); j++)
            {
                values_acc[i] += input_acc[j] * weights_acc[i[0]][j].w;
                std::cout << "Weights at index " << i[0] << ", " << j << ": "
                          << weights_acc[i[0]][j].w << std::endl;

                if (i[0] == getSize() - 1)
                    std::cout << "Input at index " << j << ": " << input_acc[j] << std::endl;
            }
            std::cout << "Z " << values_acc[i[0]] << " at index " << i[0] << std::endl;
            values_acc[i] += bias_acc[i];
            values_acc[i] = activation[0](values_acc[i]);
        });

//            std::cout << "Input at index " << i << ": " << input_acc[i] << std::endl;
    });
    queue.wait();
    auto values_acc = values.get_access<cl::sycl::access::mode::write>();
    auto bias_acc = const_cast<cl::sycl::buffer<double, 1>&>(biases).get_access<cl::sycl::access::mode::read>();
    for (int i = 0; i < values.get_range()[0]; i++)
        std::cout << "Values at index " << i << ": " << values_acc[i] << ", bias: " << bias_acc[i]
                  << std::endl;
    valuesSet = true;
    return values;
}

void nn::sycl::Layer::initValues()
{
    queue.submit([&](cl::sycl::handler& cgh) {
        auto values_acc = values.get_access<cl::sycl::access::mode::write>();
        cgh.parallel_for(values.get_range(), [=](cl::sycl::id<1> i) {
            values_acc[i] = 0.0;
        });
    });
}

void nn::sycl::Layer::setWeights(const std::vector<std::vector<nn::abs::Connection>>& w)
{
    cl::sycl::buffer<nn::abs::Connection, 2> b_w{cl::sycl::range<2>{w.size(), w[0].size()}};
    {
        auto w_acc = b_w.get_access<cl::sycl::access::mode::write>();
        for (int i = 0; i < w_acc.get_range()[0]; i++)
            for (int j = 0; j < w_acc.get_range()[1]; j++)
                w_acc[cl::sycl::id<2>{(unsigned long) i, (unsigned long) j}] = w[i][j];
    }
    setWeights(b_w);
}

//cl::sycl::buffer<double, 1>
//nn::sycl::Layer::calculateSycl(cl::sycl::handler& cgh) const
//{
//    {
//        auto values_acc = values.get_access<cl::sycl::access::mode::write>();
//        auto bias_acc = const_cast<cl::sycl::buffer<double, 1>&>(biases).get_access<cl::sycl::access::mode::read>();
//        auto weights_acc = connectionPreviousLayer->weights.get_access<cl::sycl::access::mode::read>();
//        auto activation = cl::sycl::buffer<nn::abs::Activation, 1>{activationFunction.get(), cl::sycl::range<1>{
//                1}}.get_access<cl::sycl::access::mode::read>();
//        auto input_acc = connectionPreviousLayer->from->calculateSycl().get_access<cl::sycl::access::mode::read>();
//        cgh.parallel_for(
//                cl::sycl::range<2>{(unsigned long) getSize(), (unsigned long) connectionPreviousLayer->from->getSize()},
//                [=](cl::sycl::id<2> i) {
//                    values_acc[i[0]] += input_acc[i[1]] * weights_acc[i].w;
//                });
//
//        cgh.parallel_for(cl::sycl::range<1>{(unsigned long) getSize()}, [=](cl::sycl::id<1> i) {
//            values_acc[i] += bias_acc[i];
//            values_acc[i] = activation[0](values_acc[i]);
//        });
//        queue.wait();
//    }
//    return values;
//}

int nn::sycl::InputLayer::getSize() const
{
    return size;
}

void nn::sycl::InputLayer::connect(const std::shared_ptr<nn::abs::Layer>& l)
{
    for (auto& neuronThisLayer : getNeurons())
        for (auto& neuronsNextLayer : l->getNeurons())
            neuronThisLayer->connect(neuronsNextLayer.get());
}

std::vector<double> nn::sycl::InputLayer::calculate() const
{
    queue.submit([&](cl::sycl::handler& cgh) {
        auto bias_acc = const_cast<cl::sycl::buffer<double, 1>*>(&biases)->get_access<cl::sycl::access::mode::read>();
        auto inputs_acc = const_cast<cl::sycl::buffer<double, 1>*>(&inputs)->get_access<cl::sycl::access::mode::read>();
        auto activation_acc = cl::sycl::buffer<nn::abs::Activation, 1>{activationFunction.get(), cl::sycl::range<1>{
                1}}.get_access<cl::sycl::access::mode::read>();
        auto values_acc = values.get_access<cl::sycl::access::mode::write>();
        cgh.parallel_for(cl::sycl::range<1>{(unsigned long) getSize()}, [=](cl::sycl::id<1> i) {
            values_acc[i] = activation_acc[0](inputs_acc[i] + bias_acc[i]);
        });
    });
    queue.wait();
    auto values_acc = values.get_access<cl::sycl::access::mode::read>();
    return std::vector<double>(values_acc.begin(), values_acc.end());
}

const std::vector<std::shared_ptr<nn::abs::Neuron>>& nn::sycl::InputLayer::getNeurons()
{
    if (!neuronsSet)
        setNeurons();
    return *(std::vector<std::shared_ptr<nn::abs::Neuron>>*) ((long) &neurons);
}

std::vector<std::shared_ptr<const nn::abs::Neuron>> nn::sycl::InputLayer::getNeurons() const
{
    if (!neuronsSet)
        setNeurons();
    return *(std::vector<std::shared_ptr<const nn::abs::Neuron>>*) (long) &neurons;
}

const std::shared_ptr<nn::abs::Neuron>& nn::sycl::InputLayer::getNeuron(int index)
{
    if (!neuronsSet)
        setNeurons();
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wreturn-stack-address"
    return neurons[index];
#pragma clang diagnostic pop
}

const std::shared_ptr<const nn::abs::Neuron>& nn::sycl::InputLayer::getNeuron(int index) const
{
    if (!neuronsSet)
        setNeurons();
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wreturn-stack-address"
    return (std::shared_ptr<const nn::abs::Neuron>) neurons[index];
#pragma clang diagnostic pop
}

void nn::sycl::InputLayer::setActivation(const std::shared_ptr<nn::abs::Activation>& f)
{
    activationFunction = f;
}

void nn::sycl::InputLayer::setBias(const std::vector<double>& bs)
{
    if (bs.size() != size)
        throw IncompatibleVectorException("The vector of biases has to have the same size as the layer.");
    auto biases_temp = cl::sycl::buffer<double, 1>{bs.data(), cl::sycl::range<1>{bs.size()}};
    biases = cl::sycl::buffer<double, 1>{cl::sycl::range<1>{bs.size()}};
    queue.submit([&](cl::sycl::handler& cgh) {
        auto biases_acc = biases.get_access<cl::sycl::access::mode::write>();
        auto temp_acc = biases_temp.get_access<cl::sycl::access::mode::read>();
        cgh.parallel_for(biases_acc.get_range(), [=](cl::sycl::id<1> i) {
            biases_acc[i] = temp_acc[i];
        });
    });
    queue.wait();
}

void nn::sycl::InputLayer::setWeights(const std::vector<std::map<nn::abs::Neuron*, double>>& w)
{

}

void nn::sycl::InputLayer::resetCaches() const
{
    valuesSet = false;
}

nn::sycl::InputLayer::InputLayer(int numberOfNeurons) : nn::sycl::Layer(numberOfNeurons)
{
}

void nn::sycl::InputLayer::setNeurons() const
{
    setUpNeuronsVector();
    for (int i = 0; i < size; i++)
        setNeuronsIndex(i);

    neuronsSet = true;
    for (int i = 0; i < size; i++)
        setNeuronsConnectionIndex(i);
}

void nn::sycl::InputLayer::setUpNeuronsVector() const
{
    nn::sycl::Layer::setUpNeuronsVector();
}

void nn::sycl::InputLayer::setNeuronsIndex(int i) const
{
    neurons.emplace_back(std::make_shared<nn::InputNeuron>());
    setNeuronsBiasIndex(i);
    neurons[i]->setActivation(activationFunction);
}

void nn::sycl::InputLayer::setNeuronsBiasIndex(int i) const
{
    auto& b = const_cast<trisycl::buffer<double, 1>&>(biases);
    queue.submit([&](trisycl::handler& cgh) {
        neurons[i]->setB(b.get_access<trisycl::access::mode::read>()[i]);
    });
    queue.wait();
}

nn::sycl::InputLayer::InputLayer(int numberOfNeurons, const std::shared_ptr<nn::abs::Activation>& f) : nn::sycl::Layer(
        numberOfNeurons, f)
{
}

void
nn::sycl::InputLayer::setConnectionPreviousLayer(
        const std::shared_ptr<nn::sycl::abs::Connection>& connectionPreviousLayer)
{
}

void nn::sycl::InputLayer::setConnectionNextLayer(const std::shared_ptr<nn::sycl::abs::Connection>& connectionNextLayer)
{
    this->connectionNextLayer = connectionNextLayer;
}

void nn::sycl::InputLayer::connect(const std::shared_ptr<nn::sycl::abs::Layer>& l)
{
    std::shared_ptr<nn::sycl::abs::Connection> c = std::make_shared<nn::sycl::abs::Connection>(
            static_cast<nn::sycl::Layer*>(this), l.get());
    setConnectionNextLayer(c);
    l->setConnectionPreviousLayer(c);
}

void nn::sycl::InputLayer::setNeuronsConnectionIndex(int i) const
{
    if (connectionNextLayer != nullptr)
        for (int j = 0; j < connectionNextLayer->to->getSize(); j++)
            neurons[i]->connect(connectionNextLayer->to->getNeuron(j).get());
}

const std::vector<std::shared_ptr<nn::abs::InputNeuron>>& nn::sycl::InputLayer::getInputNeurons()
{
    return *(std::vector<std::shared_ptr<nn::abs::InputNeuron>>*) ((long) &getNeurons());
}

void nn::sycl::InputLayer::setValues(const std::vector<double>& v)
{
    inputs = cl::sycl::buffer<double, 1>{cl::sycl::range<1>{(unsigned long) getSize()}};
    auto inputs_acc = inputs.get_access<cl::sycl::access::mode::write>();
    for (int i = 0; i < getSize(); i++)
        inputs_acc[i] = v[i];

    inputsSet = true;
}

void nn::sycl::InputLayer::setWeights(const cl::sycl::buffer<nn::abs::Connection, 2>& w)
{
    checkForErrorsSetWeightsBuffer(w);
    connectionNextLayer->weights = w;
}

void nn::sycl::InputLayer::checkForErrorsSetWeightsBuffer(const cl::sycl::buffer<nn::abs::Connection, 2>& w) const
{
    if (!this->connectionNextLayer)
        throw nn::sycl::Layer::NoConnectionException(
                "Connection to a next layer must be specified to execute this function");
    if (w.get_range()[1] != this->getSize() || w.get_range()[0] != this->connectionNextLayer->to->getSize())
        throw nn::sycl::Layer::IncompatibleVectorException(
                "Given buffer must have same size in first dimension as the next layers size and the second dimension must be equal to the size of the current layer");
}

cl::sycl::buffer<double, 1>
nn::sycl::InputLayer::calculateSycl() const
{
    queue.submit([&](cl::sycl::handler& cgh) {
        auto bias_acc = const_cast<cl::sycl::buffer<double, 1>&>(biases).get_access<cl::sycl::access::mode::read>();
        auto inputs_acc = const_cast<cl::sycl::buffer<double, 1>&>(inputs).get_access<cl::sycl::access::mode::read>();
        auto activation_acc = cl::sycl::buffer<nn::abs::Activation, 1>{activationFunction.get(), cl::sycl::range<1>{
                1}}.get_access<cl::sycl::access::mode::read>();
        auto values_acc = values.get_access<cl::sycl::access::mode::write>();
        cgh.parallel_for(cl::sycl::range<1>{(unsigned long) getSize()}, [=](cl::sycl::id<1> i) {
            values_acc[i] = activation_acc[0](inputs_acc[i] + bias_acc[i]);
            std::cout << "Values at index " << i[0] << ": " << values_acc[i] << " and bias: " << bias_acc[i]
                      << std::endl;
        });
    });
    queue.wait();
    return values;
}

void nn::sycl::InputLayer::setWeights(const std::vector<std::vector<nn::abs::Connection>>& w)
{
    nn::sycl::Layer::setWeights(w);
}

//cl::sycl::buffer<double, 1>
//nn::sycl::InputLayer::calculateSycl(cl::sycl::handler& cgh) const
//{
//    {
//        auto bias_acc = const_cast<cl::sycl::buffer<double, 1>*>(&biases)->get_access<cl::sycl::access::mode::read>();
//        auto inputs_acc = const_cast<cl::sycl::buffer<double, 1>*>(&inputs)->get_access<cl::sycl::access::mode::read>();
//        auto activation_acc = cl::sycl::buffer<nn::abs::Activation, 1>{activationFunction.get(), cl::sycl::range<1>{
//                1}}.get_access<cl::sycl::access::mode::read>();
//        auto values_acc = values.get_access<cl::sycl::access::mode::write>();
//        cgh.parallel_for(cl::sycl::range<1>{(unsigned long) getSize()}, [=](cl::sycl::id<1> i) {
//            values_acc[i] = activation_acc[0](inputs_acc[i] + bias_acc[i]);
//        });
//        queue.wait();
//    }
//    return values;
//}
