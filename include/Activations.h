#ifndef NEURONAL_NETWORK_ACTIVATIONS_H
#define NEURONAL_NETWORK_ACTIVATIONS_H

#include "pch.h"
#include "abstract/Activation.h"

namespace nn::activations
{
    class Sigmoid : public nn::abs::Activation
    {
    public:
        double operator()(double z) const override
        {
            return 1 / (1 + exp(-z));
        }

        double derivative(double z) const override
        {
            return 1 / (4 * pow(cosh(z / 2), 2));
        }
    };

    class Tanh : public nn::abs::Activation
    {
    public:
        double operator()(double z) const override
        {
            return tanh(z);
        }

        double derivative(double z) const override
        {
            return 1 / pow(cosh(z), 2);
        }
    };

    class Linear : public nn::abs::Activation
    {
    public:
        double operator()(double z) const override
        {
            return z;
        }

        double derivative(double z) const override
        {
            return 1;
        }
    };

    class Relu : public nn::abs::Activation
    {
    public:
        double operator()(double z) const override
        {
            return fmax(0.0, z);
        }

        double derivative(double z) const override
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

        double operator()(double z) const override
        {
            return fmax(alpha * z, z);
        }

        double derivative(double z) const override
        {
            return z > 0.0 ? 1.0 : alpha;
        }
    };

    class Elu : public nn::abs::Activation
    {
    public:
        double alpha;

        Elu(double alpha = 1.0) : alpha(alpha)
        {

        }

        double operator()(double z) const override
        {

            return z > 0 ? z : alpha * (exp(z) - 1);
        }

        double derivative(double z) const override
        {
            return z > 0.0 ? 1.0 : alpha * exp(z);
        }
    };
}

#endif //NEURONAL_NETWORK_ACTIVATIONS_H
