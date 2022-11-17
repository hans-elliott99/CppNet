#include <iostream>
#include <cmath>
#include <vector>
#include <algorithm>


#pragma once

float sigmoid(float x);
float sigmoid_derivative(float x);

float relu(float x);
float relu_derivative(float x);

template <typename T>
T random(T low, T high)
{
    return low + static_cast<T>(rand()) / ( static_cast<T>(RAND_MAX / (high - low)) );
}

// float random(float low, float high);
float euclidean_distance(const std::vector<float>& vec1,  const std::vector<float>& vec2);