#include <Neural_Network.h>

int main()
{
//    auto _tp1 = std::chrono::high_resolution_clock::now();
    std::vector<std::shared_ptr<nn::sycl::Layer>> layers;
    auto inputLayer = std::make_shared<nn::sycl::InputLayer>(284, std::make_shared<nn::activations::Sigmoid>());
    layers.reserve(20);
    for (int i = 0; i < 20; i++)
        layers.push_back(std::make_shared<nn::sycl::Layer>(284, std::make_shared<nn::activations::Sigmoid>()));
    layers.push_back(std::make_shared<nn::sycl::Layer>(16, std::make_shared<nn::activations::Sigmoid>()));
    layers.push_back(std::make_shared<nn::sycl::Layer>(16, std::make_shared<nn::activations::Sigmoid>()));
    layers.push_back(std::make_shared<nn::sycl::Layer>(9, std::make_shared<nn::activations::Sigmoid>()));
    inputLayer->connect(static_pointer_cast<nn::sycl::abs::Layer>(layers[0]));
    for (int i = 0; i < layers.size() - 1; i++)
        layers[i]->connect(static_pointer_cast<nn::sycl::abs::Layer>(layers[i + 1]));
    std::vector<double> inputVector(284);
    for (auto& e : inputVector)
    {
        int r1 = rand();
        int r2 = rand();
        long r = r1 << 31 | r2;
        e = *(double*) &r;
    }

    int iterations = 10000;
//    std::vector<std::chrono::duration<double>> times(iterations);
    for (int i = 0; i < iterations; i++)
    {
//        auto tp1 = std::chrono::high_resolution_clock::now();
        inputLayer->setValues(inputVector);
        std::vector<double> resultCalc = layers[layers.size() - 1]->calculate();
//        auto tp2 = std::chrono::high_resolution_clock::now();
//        std::chrono::duration<double> elapsedTime = tp2 - tp1;
//        std::cout << "Time spend on net.calculate: " << elapsedTime.count() * 1000 << " ms" << std::endl;
//        times.push_back(elapsedTime);
    }

//    double addedTime = 0;
//    for (auto& t : times)
//        addedTime += t.count() * 1000;

//    std::cout << "average time: " << addedTime / iterations << " ms" << std::endl;
}

