#ifndef NEURONAL_NETWORK_TEST_NETWORK_H
#define NEURONAL_NETWORK_TEST_NETWORK_H

#define MessureTimeNeededToCalculate

#include "Neural_Network.h"

namespace nn::test
{
    class Network : public testing::Test
    {
    public:
        int initialNumberOfLayers = 5;
        nn::Network network{initialNumberOfLayers};
    };
}

#endif //NEURONAL_NETWORK_NETWORK_H
