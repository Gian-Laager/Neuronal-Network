#ifndef  NEURONAL_NETWORK_SYCL_ABSTRACT_LAYER_H
#define NEURONAL_NETWORK_SYCL_ABSTRACT_LAYER_H

#include "abstract/Layer.h"

namespace nn::sycl::abs
{
    class Connection;

    class Layer : public nn::abs::Layer
    {
    public:
        virtual void
        setConnectionPreviousLayer(const std::shared_ptr<nn::sycl::abs::Connection>& connectionPreviousLayer) = 0;

        virtual void setConnectionNextLayer(const std::shared_ptr<nn::sycl::abs::Connection>& connectionNextLayer) = 0;

        virtual void connect(const std::shared_ptr<nn::sycl::abs::Layer>& l) = 0;

        virtual void setWeights(const cl::sycl::buffer<nn::abs::Connection, 2>& w) = 0;

        virtual void setWeights(const std::vector<std::vector<nn::abs::Connection>>& w) = 0;

        virtual cl::sycl::buffer<double, 1> calculateSycl() const = 0;

//        virtual cl::sycl::buffer<double, 1>
//        calculateSycl(cl::sycl::handler& cgh) const = 0;
    };

    class InputLayer : public nn::abs::InputLayer, public nn::sycl::abs::Layer
    {
    };

    class Connection
    {
    public:
        cl::sycl::buffer<nn::abs::Connection, 2> weights;
        nn::sycl::abs::Layer* from = nullptr;
        nn::sycl::abs::Layer* to = nullptr;

        Connection(nn::sycl::abs::Layer* from, nn::sycl::abs::Layer* to) : from(from),
                                                                           to(to)
        {
            weights = cl::sycl::buffer<nn::abs::Connection, 2>{
                    cl::sycl::range<2>{(unsigned long) from->getSize(), (unsigned long) to->getSize()}};
        }

        Connection() = default;
    };
}

#endif // NEURONAL_NETWORK_SYCL_ABSTRACT_LAYER_H
