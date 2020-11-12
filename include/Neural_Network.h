#ifndef NEURONAL_NETWORK_NEURAL_NETWORK_H
#define NEURONAL_NETWORK_NEURAL_NETWORK_H

#include "abstract.h"

#include "Neuron.h"
#include "Layer.h"
#include "Network.h"
#include "Activations.h"

namespace nn
{
    double derivative(double x, double dx, std::function<double(double)> func);
}

#endif //NEURONAL_NETWORK_NEURAL_NETWORK_H
