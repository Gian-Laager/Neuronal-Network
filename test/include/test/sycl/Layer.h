#ifndef NEURONAL_NETWORK_TEST_SYCL_LAYER_H
#define NEURONAL_NETWORK_TEST_SYCL_LAYER_H

#include "test/pch.h"
#include <sycl/Layer.h>

namespace nn::sycl::test
{
    class Layer : public testing::Test
    {
        nn::sycl::Layer layer;
    };
}

#endif //NEURONAL_NETWORK_TEST_SYCL_LAYER_H
