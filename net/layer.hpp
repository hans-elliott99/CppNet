#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include "../matrix/matrix.hpp"
#include "../matrix/matrix.tpp"
#include "functions.hpp"

#pragma once

#define FIXED_DOUBL(x) std::fixed << std::setprecision(6) << (x)


// namespace nn
// refactor so linear is like a base class?

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

class ReLU
{   
public:
    Matrix<float> weight;
    Matrix<float> bias;
    Matrix<float> Wgrad;
    Matrix<float> Bgrad;
    Matrix<float> relu_inputs; //save for calculating derivative.

    bool use_bias = true;

public:
    ReLU(size_t n_in, size_t n_out, bool use_bias = true);
    ~ReLU() {};

    Matrix<float> forward(const Matrix<float>& X);
    Matrix<float> backward(const Matrix<float>& X, const Matrix<float>& grad_flow);

};