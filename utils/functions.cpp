#include "functions.h"


// Sigmoid
double sigmoid(double x) 
{
    return 1.0 / (1.0 + exp(-x));
}

double sigmoid_derivative(double x)
{
    return sigmoid(x) * (1 - sigmoid(x));
}

// ReLU
double relu(double x)
{
    if (x > 0)
        {return x; }
    else
        {return 0; }
}

double relu_derivative(double x)
{
    if (x > 0)
        {return 1; }
    else
        {return 0; }
}

// Random Double Generator
double random(double low, double high)
{
    return low + static_cast<double> (rand()) / ( static_cast<double> (RAND_MAX / (high - low)) );
}

// Euclidean Distance
double euclidean_distance(const std::vector<double>& vec1,  const std::vector<double>& vec2)
{
    double dist {0};
    for (size_t idx {0}; idx < vec1.size(); idx++)
        { dist += pow(vec1[idx] - vec2[idx], 2); }

    return pow(dist, 0.5);
}



// int main()
// {
//     std::vector<double> v1 = {1., 2., 3.};
//     std::vector<double> v2 = {4., 5., 6.};

//     std::cout << euclidean_distance(v1, v2);
    

//     // std::cout << random(0.0, 1.0);
// }

