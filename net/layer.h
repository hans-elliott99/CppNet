#include <iostream>
#include <vector>
#include <unordered_map>
#include <cmath>
#include "../matrix/matrix.h"
#include "../matrix/matrix.cpp"


// namespace nn

class Linear
{   
public:
    Matrix<float> weight;
    Matrix<float> bias;
    Matrix<float> grad;
    bool use_bias;

public:
    Linear(size_t n_in, size_t n_out, bool use_bias = true);
    ~Linear() {};

    Matrix<float> forward(Matrix<float> X);



};