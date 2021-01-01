#ifndef NEURONAL_NETWORK_NEURAL_NETWORK_H
#define NEURONAL_NETWORK_NEURAL_NETWORK_H

#include "abstract.h"

#include "Neuron.h"
#include "Layer.h"
#include "Network.h"
#include "Activations.h"
#include "LossFunctions.h"
#include "Backpropagator.h"

#ifdef NEURONAL_NETWORK_USE_SYCL
#include "sycl/Layer.h"
#endif

#endif //NEURONAL_NETWORK_NEURAL_NETWORK_H
