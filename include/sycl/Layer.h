#ifndef NEURONAL_NETWORK_SYCL_LAYER_H
#define NEURONAL_NETWORK_SYCL_LAYER_H

#include "abstract/Layer.h"
#include "Neuron.h"
#include "sycl/abstract/Layer.h"

namespace nn::sycl
{

    class Layer : public nn::sycl::abs::Layer
    {
    protected:
        int size = 0;
        cl::sycl::buffer<double, 1> biases;
        mutable bool neuronsSet = false;
        mutable std::vector<std::shared_ptr<nn::abs::Neuron>> neurons;
        mutable bool valuesSet = false;
        mutable cl::sycl::buffer<double, 1> values;
        mutable cl::sycl::queue queue;
        std::shared_ptr<nn::abs::Activation> activationFunction = std::make_shared<nn::activations::Linear>();

        std::shared_ptr<nn::sycl::abs::Connection> connectionPreviousLayer = nullptr;

        std::shared_ptr<nn::sycl::abs::Connection> connectionNextLayer = nullptr;

        virtual void setNeurons() const;

        virtual void setNeuronsIndex(int i) const;

        virtual void setNeuronsBiasIndex(int i) const;

        virtual void setUpNeuronsVector() const;

        virtual void setNeuronsConnectionIndex(int i) const;

        virtual void checkForErrorsSetWeightsBuffer(const cl::sycl::buffer<nn::abs::Connection, 2>& w) const;

        std::vector<double> calculateWithNonSyclNeurons() const;

        std::vector<double> calculateSyclWithVectorReturn() const;

        std::map<nn::abs::Neuron*, double> getWeightsMap(int index) const;

        void initValues();
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

        cl::sycl::buffer<double, 1>
        calculateSycl() const override;

        void setWeights(const std::vector<std::vector<nn::abs::Connection>>& w) override;

        EXCEPTION(IncompatibleVectorException);

        EXCEPTION(NoConnectionException);
    };

    class InputLayer : public nn::sycl::Layer, public nn::sycl::abs::InputLayer
    {
    protected:
        mutable bool inputsSet = false;
        cl::sycl::buffer<double, 1> inputs;

        void setNeurons() const override;

        void setNeuronsIndex(int i) const override;

        void setNeuronsBiasIndex(int i) const override;

        void setUpNeuronsVector() const override;

        void setNeuronsConnectionIndex(int i) const override;

        void checkForErrorsSetWeightsBuffer(const cl::sycl::buffer<nn::abs::Connection, 2>& w) const override;

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

        cl::sycl::buffer<double, 1>
        calculateSycl() const override;

        void setWeights(const std::vector<std::vector<nn::abs::Connection>>& w) override;

//        cl::sycl::buffer<double, 1>
//        calculateSycl(cl::sycl::handler& cgh) const override;

        EXCEPTION(IncompatibleVectorException);
    };
}

#endif //NEURONAL_NETWORK_SYCL_LAYER_H
