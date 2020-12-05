#ifndef NEURONAL_NETWORK_TEST_SYCL_LAYER_H
#define NEURONAL_NETWORK_TEST_SYCL_LAYER_H

#include "test/pch.h"
#include <sycl/Layer.h>
#include <Layer.h>

#define TEST_PF(package_name, test_fixture, test_name) TEST_F(package_name##_##test_fixture, test_name)

#define TEST_PACKAGE_FIXTURE(package_name, test_fixture) class package_name##_##test_fixture

namespace nn::sycl::test
{
    TEST_PACKAGE_FIXTURE(Sycl, Layer) : public testing::Test
    {
    public:
        int numberOfNeurons = 3;
        nn::sycl::Layer layer;

        void SetUp() override;
    };
}

#endif //NEURONAL_NETWORK_TEST_SYCL_LAYER_H
