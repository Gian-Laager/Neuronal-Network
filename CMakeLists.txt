cmake_minimum_required(VERSION 3.17)
project(Neuronal_Network)

set(CMAKE_CXX_STANDARD 20)

set(SRC
        src/Neural_Network.cpp
        src/Neuron.cpp
        src/Layer.cpp
        src/Network.cpp
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
)

set(PCH
        include/pch.h
)

include_directories(
        deps/SYCL/include
        include/
        src/
)

add_library(${PROJECT_NAME} STATIC ${SRC} ${HEAD})

target_precompile_headers(${PROJECT_NAME} PUBLIC ${PCH})

set(TEST_SRC
        test/src/test/main.cpp
        test/src/test/Neuron.cpp
        test/src/test/Layer.cpp
        test/src/test/Network.cpp
)

set(TEST_HEAD
        test/include/test/Neuron.h
        test/include/test/Layer.h
        test/include/test/Network.h
)

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
