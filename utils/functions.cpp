#include "functions.h"


// Sigmoid
float sigmoid(float x) 
{
    return 1.0f / (1.0f + exp(-x));
}

float sigmoid_derivative(float x)
{
    return sigmoid(x) * (1.0f - sigmoid(x));
}

// ReLU
float relu(float x)
{
    if (x > 0)
        {return x; }
    else
        {return 0; }
}

float relu_derivative(float x)
{
    if (x > 0)
        {return 1; }
    else
        {return 0; }
}

// Random Double Generator
template <typename T>
T random(int low, int high)
{
    return low + static_cast<T>(rand()) / ( static_cast<T>(RAND_MAX / (high - low)) );
}

// Euclidean Distance
float euclidean_distance(const std::vector<float>& vec1,  const std::vector<float>& vec2)
{
    float dist {0};
    for (size_t idx {0}; idx < vec1.size(); idx++)
        { dist += pow(vec1[idx] - vec2[idx], 2); }

    return pow(dist, 0.5);
}
