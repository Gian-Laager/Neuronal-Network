#include <Neural_Network.h>

int main()
{
        auto complexNetwork = nn::Network{3};
        complexNetwork.pushLayer(
                std::make_shared<nn::InputLayer<nn::InputNeuron>>(2, std::make_shared<nn::activations::Sigmoid>()));
        complexNetwork.pushLayer(
                std::make_shared<nn::Layer<nn::Neuron>>(3, std::make_shared<nn::activations::Sigmoid>()));
        complexNetwork.pushLayer(std::make_shared<nn::Layer<nn::Neuron>>(2, std::make_shared<nn::activations::Tanh>()));

        std::vector<std::vector<double>> xs = {{0.25, 2.0},
                                               {0.75, 4.0},
                                               {1.5,  6.0}};
        std::vector<std::vector<double>> ys = {{0.0, 0.2},
                                               {1.0, 0.4},
                                               {1.0, 0.6}};

        double learningRate = 0.5e-5;

        complexNetwork.initializeFitting(std::make_shared<nn::losses::MSE>());
        complexNetwork.fit(xs, ys, learningRate, 3000);
}

