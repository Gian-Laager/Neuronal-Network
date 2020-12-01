#include "sycl/Layer.h"

int nn::sycl::Layer::getSize() const
{
    return 0;
}

void nn::sycl::Layer::connect(const std::shared_ptr<nn::abs::Layer>& l)
{

}

std::vector<double> nn::sycl::Layer::calculate() const
{
    return std::vector<double>();
}

const std::vector<std::shared_ptr<nn::abs::Neuron>>& nn::sycl::Layer::getNeurons()
{
    return std::vector<std::shared_ptr<nn::abs::Neuron>>();
}

std::vector<std::shared_ptr<const nn::abs::Neuron>> nn::sycl::Layer::getNeurons() const
{
    return std::vector<std::shared_ptr<const nn::abs::Neuron>>();
}

const std::shared_ptr<nn::abs::Neuron>& nn::sycl::Layer::getNeuron(int index)
{
    return std::shared_ptr<nn::abs::Neuron>();
}

const std::shared_ptr<const nn::abs::Neuron>& nn::sycl::Layer::getNeuron(int index) const
{
    return std::shared_ptr<const nn::abs::Neuron>();
}

void nn::sycl::Layer::setActivation(const std::shared_ptr<nn::abs::Activation>& f)
{

}

void nn::sycl::Layer::setBias(const std::vector<double>& bs)
{

}

void nn::sycl::Layer::setWeights(const std::vector<std::map<nn::abs::Neuron*, double>>& weights)
{

}

void nn::sycl::Layer::resetCaches() const
{

}
