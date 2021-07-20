#!/bin/bash
#*******************************************************************
# This source code was derived from the LinkedIn course
# Training Neural Networks in C++ by Eduardo Corpeno
# 
# This was updated to run on a MacBook Pro
#*******************************************************************

#*******************************************************************
# Compile the source code on a MacBook Pro using MacOS v11.4 and g++
#*******************************************************************
g++ -std=c++17 neural_networks.cc mlp.cc -o neural_network

#*******************************************************************
# Execute the output binary
#*******************************************************************
./neural_network
