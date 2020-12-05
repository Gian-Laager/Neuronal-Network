#ifndef NEURONAL_NETWORK_SYCL_LAYER_H
#define NEURONAL_NETWORK_SYCL_LAYER_H

#include "abstract/Layer.h"
#include "Neuron.h"
#include "sycl/abstract/Layer.h"

namespace nn::sycl
{

    class Layer : public nn::sycl::abs::Layer
    {
    private:
        int size = 0;
        cl::sycl::buffer<double, 1> biases;
        mutable bool neuronsSet = false;
        mutable std::vector<std::shared_ptr<nn::abs::Neuron>> neurons;
        mutable cl::sycl::queue queue;
        std::shared_ptr<nn::abs::Activation> activationFunction = std::make_shared<nn::activations::Linear>();

        std::shared_ptr<nn::sycl::abs::Connection> connectionPreviousLayer = nullptr;

        std::shared_ptr<nn::sycl::abs::Connection> connectionNextLayer = nullptr;

        void setNeurons() const;

        void setNeuronsIndex(int i) const;

        void setNeuronsBiasIndex(int i) const;

        void setUpNeuronsVector() const;

        void setNeuronsConnectionIndex(int i) const;

        void checkForErrorsSetWeightsBuffer(const cl::sycl::buffer<nn::abs::Connection, 2>& w) const;

    public:
        Layer(int numberOfNeurons);

        Layer(int numberOfNeurons, const std::shared_ptr<nn::abs::Activation>& f);

        Layer() = default;

        int getSize() const override;

        void connect(const std::shared_ptr<nn::abs::Layer>& l) override;

        void connect(const std::shared_ptr<nn::sycl::abs::Layer>& l) override;

        std::vector<double> calculate() const override;

        const std::vector<std::shared_ptr<nn::abs::Neuron>>& getNeurons() override;

        std::vector<std::shared_ptr<const nn::abs::Neuron>> getNeurons() const override;

        const std::shared_ptr<nn::abs::Neuron>& getNeuron(int index) override;

        const std::shared_ptr<const nn::abs::Neuron>& getNeuron(int index) const override;

        void setActivation(const std::shared_ptr<nn::abs::Activation>& f) override;

        void setBias(const std::vector<double>& bs) override;

        void setWeights(const std::vector<std::map<nn::abs::Neuron*, double>>& w) override;

        void resetCaches() const override;

        void setConnectionNextLayer(const std::shared_ptr<nn::sycl::abs::Connection>& connectionNextLayer) override;

        void
        setConnectionPreviousLayer(const std::shared_ptr<nn::sycl::abs::Connection>& connectionPreviousLayer) override;

        void setWeights(const cl::sycl::buffer<nn::abs::Connection, 2>& w) override;

        EXCEPTION(IncompatibleVectorException);

        EXCEPTION(NoConnectionException);
    };

    class InputLayer : public nn::sycl::abs::InputLayer
    {
    private:
        int size = 0;
        cl::sycl::buffer<double, 1> biases;
        mutable bool neuronsSet = false;
        mutable std::vector<std::shared_ptr<nn::abs::InputNeuron>> neurons;
        mutable cl::sycl::queue queue;
        std::shared_ptr<nn::abs::Activation> activationFunction = std::make_shared<nn::activations::Linear>();

        std::shared_ptr<nn::sycl::abs::Connection> connectionNextLayer = nullptr;
        mutable bool valuesSet = false;
        mutable cl::sycl::buffer<double, 1> values;
        mutable bool inputsSet = false;
        cl::sycl::buffer<double, 1> inputs;

        void setNeurons() const;

        void setNeuronsIndex(int i) const;

        void setNeuronsBiasIndex(int i) const;

        void setUpNeuronsVector() const;

        void setNeuronsConnectionIndex(int i) const;

        void checkForErrorsSetWeightsBuffer(const cl::sycl::buffer<nn::abs::Connection, 2>& w) const;
    public:
        InputLayer(int numberOfNeurons);

        InputLayer(int numberOfNeurons, const std::shared_ptr<nn::abs::Activation>& f);

        InputLayer() = default;

        int getSize() const override;

        void connect(const std::shared_ptr<nn::abs::Layer>& l) override;

        void connect(const std::shared_ptr<nn::sycl::abs::Layer>& l) override;

        std::vector<double> calculate() const override;

        const std::vector<std::shared_ptr<nn::abs::Neuron>>& getNeurons() override;

        std::vector<std::shared_ptr<const nn::abs::Neuron>> getNeurons() const override;

        const std::shared_ptr<nn::abs::Neuron>& getNeuron(int index) override;

        const std::shared_ptr<const nn::abs::Neuron>& getNeuron(int index) const override;

        void setActivation(const std::shared_ptr<nn::abs::Activation>& f) override;

        void setBias(const std::vector<double>& bs) override;

        void setWeights(const std::vector<std::map<nn::abs::Neuron*, double>>& w) override;

        void resetCaches() const override;

        void setConnectionNextLayer(const std::shared_ptr<nn::sycl::abs::Connection>& connectionNextLayer) override;

        void
        setConnectionPreviousLayer(const std::shared_ptr<nn::sycl::abs::Connection>& connectionPreviousLayer) override;

        void setValues(const std::vector<double>& v) override;

        const std::vector<std::shared_ptr<nn::abs::InputNeuron>>& getInputNeurons() override;

        void setWeights(const cl::sycl::buffer<nn::abs::Connection, 2>& w) override;

        EXCEPTION(IncompatibleVectorException);
    };
}

#endif //NEURONAL_NETWORK_SYCL_LAYER_H
