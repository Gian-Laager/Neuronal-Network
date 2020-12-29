#include <Neural_Network.h>

int main()
{
    cl::sycl::queue queue;
    nn::Network network{3};
    network.pushLayer(std::make_shared<nn::InputLayer<nn::InputNeuron>>(2, std::make_shared<nn::activations::Sigmoid>()));
    network.pushLayer(std::make_shared<nn::Layer<nn::Neuron>>(3, std::make_shared<nn::activations::Relu>()));
    network.pushLayer(std::make_shared<nn::Layer<nn::Neuron>>(2, std::make_shared<nn::activations::Tanh>()));

    network.setBias(0, std::vector<double>{0.2, -3});
    network.setWeights(0, std::vector<std::map<nn::abs::Neuron*, double>>{
            std::map<nn::abs::Neuron*, double>{{network.getLayer(1)->getNeuron(0).get(), 0.75},
                                               {network.getLayer(1)->getNeuron(1).get(), 0.5},
                                               {network.getLayer(1)->getNeuron(2).get(), 0.25}},

            std::map<nn::abs::Neuron*, double>{{network.getLayer(1)->getNeuron(0).get(), 0.3},
                                               {network.getLayer(1)->getNeuron(1).get(), 0.2},
                                               {network.getLayer(1)->getNeuron(2).get(), 0.1}}});

    network.setBias(1, std::vector<double>{0.8, 0.3, 0.2});
    network.setWeights(1, std::vector<std::map<nn::abs::Neuron*, double>>{
            std::map<nn::abs::Neuron*, double>{{network.getLayer(2)->getNeuron(0).get(), 0.2},
                                               {network.getLayer(2)->getNeuron(1).get(), 0.4}},

            std::map<nn::abs::Neuron*, double>{{network.getLayer(2)->getNeuron(0).get(), 0.6},
                                               {network.getLayer(2)->getNeuron(1).get(), 0.8}},

            std::map<nn::abs::Neuron*, double>{{network.getLayer(2)->getNeuron(0).get(), 1},
                                               {network.getLayer(2)->getNeuron(1).get(), 1.2}}});

    network.setBias(2, std::vector<double>{-1.2, -0.1});

    network.setInputs(std::vector{0.5, 0.25});

    for (auto n : network.calculate())
        std::cout << n << std::endl;

    return 0;
}

