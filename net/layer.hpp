#include <iostream>
#include <vector>
#include <unordered_map>
#include <cmath>
#include "../matrix/matrix.hpp"
#include "../matrix/matrix.tpp"
#include "../utils/functions.hpp"

#pragma once
// namespace nn

class Linear
{   
public:
    Matrix<float> weight;
    Matrix<float> bias;
    Matrix<float> Wgrad;
    Matrix<float> Bgrad;
    bool use_bias = true;

public:
    Linear(size_t n_in, size_t n_out, bool use_bias = true);
    ~Linear() {};

    Matrix<float> forward(const Matrix<float>& X);
    Matrix<float> backward(const Matrix<float>& X, const Matrix<float>& grad_flow);

};