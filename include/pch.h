#ifndef NEURONAL_NETWORK_PCH_H
#define NEURONAL_NETWORK_PCH_H

#define EXCEPTION(Name) class Name : public std::exception\
                        {\
                        public:\
                            std::string message;              \
                            Name(std::string message) : message(std::move(message)) {}       \
                            \
                            const char* what() const noexcept override { return message.c_str(); }              \
                        };

#include <vector>
#include <map>
#include <string>
#include <utility>
#include <memory>
#include <functional>
#include <math.h>
#include <iostream>

#endif //NEURONAL_NETWORK_PCH_H
