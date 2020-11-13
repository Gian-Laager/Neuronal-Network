#ifndef NEURONAL_NETWORK_ACTIVATIONS_H
#define NEURONAL_NETWORK_ACTIVATIONS_H

#include "pch.h"
#include "abstract/Activation.h"

namespace nn::activations
{
    class Sigmoid : public nn::abs::Activation
    {
    public:
        double operator()(double z) override
        {
            return 1 / (1 + exp(-z));
        }

        double derivative(double z) override
        {
            return this->operator()(z) * (1 - this->operator()(z));
        }
    };

    class Tanh : public nn::abs::Activation
    {
    public:
        double operator()(double z) override
        {
            return tanh(z);
        }

        double derivative(double z) override
        {
            return 1 - pow((exp(z) - exp(-z)) / (exp(z) + exp(-z)), 2);
        }
    };

    class Linear : public nn::abs::Activation
    {
    public:
        double operator()(double z) override
        {
            return z;
        }

        double derivative(double z) override
        {
            return 1;
        }
    };

    class Relu : public nn::abs::Activation
    {
    public:
        double operator()(double z) override
        {
            return fmax(0.0, z);
        }

        double derivative(double z) override
        {
            return z > 0.0 ? 1.0 : 0.0;
        }
    };

    class LeakyRelu : public nn::abs::Activation
    {
    public:
        double alpha;

        LeakyRelu(double alpha = 0.01) : alpha(alpha)
        {

        }

        double operator()(double z) override
        {
            return fmax(alpha * z, z);
        }

        double derivative(double z) override
        {
            return z > 0.0 ? 1.0 : 0.0;
        }
    };

    class Elu : public nn::abs::Activation
    {
    public:
        double alpha;

        Elu(double alpha = 1.0) : alpha(alpha)
        {

        }

        double operator()(double z) override
        {

            return z > 0 ? z : alpha * (exp(z) - 1);
        }

        double derivative(double z) override
        {
            return z > 0.0 ? 1.0 : alpha * exp(z);
        }
    };
}

#endif //NEURONAL_NETWORK_ACTIVATIONS_H
