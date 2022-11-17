#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <assert.h>
#include "../matrix/matrix.hpp"
#include "../matrix/matrix.tpp"
#include "functions.hpp"

#pragma once

#define FIXED_DOUBL(x) std::fixed << std::setprecision(6) << (x)

// namespace nn
class Layer
{   
public:
    Matrix<float> weight;
    Matrix<float> bias;
    Matrix<float> Wgrad;
    Matrix<float> Bgrad;
    bool use_bias = true;

public:

    Layer(size_t n_in, size_t n_out, bool use_bias = true);

    virtual Matrix<float> forward(const Matrix<float>& X) = 0;
    virtual Matrix<float> backward(const Matrix<float>& X, const Matrix<float>& grad_flow) = 0;

protected:
    Matrix<float> _layer_forward(const Matrix<float>& X);
    Matrix<float> _layer_backward(const Matrix<float>& X, const Matrix<float>& grad_flow);
};



class Linear : public Layer
{
public:
    Linear(size_t n_in, size_t n_out, bool use_bias = true);


    Matrix<float> forward(const Matrix<float>& X);
    Matrix<float> backward(const Matrix<float>& X, const Matrix<float>& grad_flow);
};



class ReLU : public Layer
{   
private:
    Matrix<float> _activ_inputs; //save for calculating derivative.

public:
    ReLU(size_t n_in, size_t n_out, bool use_bias = true);

    Matrix<float> forward(const Matrix<float>& X);
    Matrix<float> backward(const Matrix<float>& X, const Matrix<float>& grad_flow);

};


class Sigmoid : public Layer
{
public:
    Matrix<float> _activ_outputs;

public:
    Sigmoid(size_t n_in, size_t n_out, bool use_bias = true);

    Matrix<float> forward(const Matrix<float>& X);
    Matrix<float> backward(const Matrix<float>& X, const Matrix<float>& grad_flow);

};