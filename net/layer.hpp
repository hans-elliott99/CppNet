#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <random>
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
    size_t n_in;
    size_t n_out;

public:

    Layer(size_t n_in, size_t n_out, bool use_bias = true);
    
    // template<class RNG> //https://stackoverflow.com/questions/38244877/how-to-use-stdnormal-distribution
    // void initXavierNormal(float gain, RNG &gen);
    virtual void          initXavierNormal(std::default_random_engine &gen, float gain) = 0;

    virtual Matrix<float> forward(const Matrix<float>& X) = 0;
    virtual Matrix<float> backward(const Matrix<float>& X, const Matrix<float>& grad_flow) = 0;

protected:
    Matrix<float> _layer_forward(const Matrix<float>& X);
    Matrix<float> _layer_backward(const Matrix<float>& X, const Matrix<float>& grad_flow);
    void          _xavier_normal(float gain, std::default_random_engine &gen);

};


///
class Linear : public Layer
{
public:
    Linear(size_t n_in, size_t n_out, bool use_bias = true);

    void initXavierNormal(std::default_random_engine &gen, float gain = 1) {_xavier_normal(gain, gen); };

    Matrix<float> forward(const Matrix<float>& X);
    Matrix<float> backward(const Matrix<float>& X, const Matrix<float>& grad_flow);
};


///
class ReLU : public Layer
{   
private:
    Matrix<float> _activ_inputs; //save for calculating derivative.

public:
    ReLU(size_t n_in, size_t n_out, bool use_bias = true);

    void initXavierNormal(std::default_random_engine &gen, float gain = 1.4142135623730951) {_xavier_normal(gain, gen); };  //gain = sqrt(2)

    Matrix<float> forward(const Matrix<float>& X);
    Matrix<float> backward(const Matrix<float>& X, const Matrix<float>& grad_flow);
};

///
class Sigmoid : public Layer
{
public:
    Matrix<float> _activ_outputs;

public:
    Sigmoid(size_t n_in, size_t n_out, bool use_bias = true);

    void initXavierNormal(std::default_random_engine &gen, float gain = 1) {_xavier_normal(gain, gen); };

    Matrix<float> forward(const Matrix<float>& X);
    Matrix<float> backward(const Matrix<float>& X, const Matrix<float>& grad_flow);
};