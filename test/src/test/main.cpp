#include "test/pch.h"

#include "Neural_Network.h"

void performanceTest()
{
//    auto _tp1 = std::chrono::high_resolution_clock::now();
    nn::Network net{3};
    net.pushLayer(std::make_shared<nn::InputLayer<nn::InputNeuron>>(284, [](double z) -> double { return 1 / (1 - exp(-z)); }));
    net.pushLayer(std::make_shared<nn::Layer<nn::Neuron>>(16, [](double z) -> double { return 1 / (1-exp(-z)); }));
    net.pushLayer(std::make_shared<nn::Layer<nn::Neuron>>(16, [](double z) -> double { return 1 / (1-exp(-z)); }));
    net.pushLayer(std::make_shared<nn::Layer<nn::Neuron>>(9, [](double z) -> double { return 1 / (1-exp(-z)); }));

    std::vector<double> inputVector(284);
    for (auto& e : inputVector)
    {
        int r1 = rand();
        int r2 = rand();
        long r = r1 << 31 | r2;
        e = *(double*) &r;
    }

    net.setInputs(inputVector);
    int iterations = 100;
    std::vector<std::chrono::duration<double>> times(iterations);
    for (int i = 0; i < iterations; i++)
    {
        auto tp1 = std::chrono::high_resolution_clock::now();
        std::vector<double> resultCalc = net.calculate();
        auto tp2 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsedTime = tp2 - tp1;
        std::cout << "Time spend on net.calculate: " << elapsedTime.count() * 1000 << " ms" << std::endl;
        times.push_back(elapsedTime);
    }

    double addedTime = 0;
    for (auto& t : times)
        addedTime += t.count() * 1000;

    std::cout << "average time: " << addedTime / iterations << " ms" << std::endl;
}

int main()
{
    performanceTest();
    testing::InitGoogleTest();
    return RUN_ALL_TESTS();
}