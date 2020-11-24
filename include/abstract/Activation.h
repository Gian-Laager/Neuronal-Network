#ifndef NEURONAL_NETWORK_ABSTRACT_ACTIVATION_H
#define NEURONAL_NETWORK_ABSTRACT_ACTIVATION_H

namespace nn::abs
{
    class Activation
    {
    public:
        virtual double operator()(double z) const = 0;

        virtual double derivative(double z) const = 0;
    };
}

#endif //NEURONAL_NETWORK_ABSTRACT_ACTIVATION_H
