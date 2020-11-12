#ifndef NEURONAL_NETWORK_ABSTRACT_ACTIVATION_H
#define NEURONAL_NETWORK_ABSTRACT_ACTIVATION_H

namespace nn::abs
{
    class Activation
    {
    public:
        virtual double operator()(double z) = 0;

        virtual double derivative(double z) = 0;
    };
}

#endif //NEURONAL_NETWORK_ABSTRACT_ACTIVATION_H
