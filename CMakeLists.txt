cmake_minimum_required(VERSION 3.17)
project(Neuronal_Network)

set(CMAKE_CXX_STANDARD 20)

set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")

option(USE_SYCL ON)

set(SRC
        src/Neural_Network.cpp
        src/Neuron.cpp
        src/Layer.cpp
        src/Network.cpp
        src/Backpropagator.cpp
)

set(HEAD
        include/Neural_Network.h
        include/abstract.h
        include/abstract/Neuron.h
        include/Neuron.h
        include/abstract/Layer.h
        include/Layer.h
        include/abstract/Network.h
        include/Network.h
        include/abstract/Backpropagator.h
        include/Backpropagator.h
        include/Activations.h
        include/abstract/Activation.h
        include/abstract/LossFunction.h
        include/LossFunctions.h
)

if (USE_SYCL)
    set(SRC ${SRC} src/SyclLayer.cpp)
    set(HEAD ${HEAD} include/SyclLayer.h)
endif()

set(PCH
        include/pch.h
)

include_directories(
        include/
        src/
)

if (USE_SYCL)
    include_directories(deps/SYCL/include)
endif()

add_library(${PROJECT_NAME} STATIC ${SRC} ${HEAD})

target_precompile_headers(${PROJECT_NAME} PUBLIC ${PCH})

set(TEST_SRC
        test/src/test/main.cpp
        test/src/test/Neuron.cpp
        test/src/test/Layer.cpp
        test/src/test/Network.cpp
        test/src/test/Backpropagator.cpp
)

set(TEST_HEAD
        test/include/test/Neuron.h
        test/include/test/Layer.h
        test/include/test/Network.h
        test/include/test/Backpropagator.h

)

if (USE_SYCL)
    set(TEST_SRC ${TEST_SRC} test/src/test/SyclLayer.cpp)
    set(TEST_HEAD ${TEST_HEAD} test/include/test/SyclLayer.h)
endif()

set(TEST_PCH
        test/include/test/pch.h
)

add_executable(${PROJECT_NAME}_TEST ${TEST_SRC} ${TEST_HEAD})

target_precompile_headers(${PROJECT_NAME}_TEST PRIVATE ${TEST_PCH})

target_include_directories(${PROJECT_NAME}_TEST PRIVATE
        test/deps/googletest/googletest/include
        test/include
        test/src
)

target_link_directories(${PROJECT_NAME}_TEST PRIVATE
        test/deps/googletest/build/lib/
)

target_link_libraries(${PROJECT_NAME}_TEST PRIVATE
        libgtest.dylib
        ${PROJECT_NAME}
)

add_executable(PerformanceTest test/src/PerformanceTest.cpp)

target_link_libraries(PerformanceTest PRIVATE
        ${PROJECT_NAME}
)

if (USE_SYCL)
    add_executable(SyclPerformanceTest test/src/SyclPerformanceTest.cpp)
    target_link_libraries(SyclPerformanceTest PRIVATE
            ${PROJECT_NAME}
    )
endif()