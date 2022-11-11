#include "layer.h"

#define FIXED_DOUBL(x) std::fixed << std::setprecision(6) << (x)

Linear::Linear(size_t n_in, size_t n_out, bool use_bias)
{
    use_bias = use_bias;
    // Initialize Weight & Gradient Matrices
    double scalar {1.0 / n_in};
           scalar = sqrt(scalar);

    weight = Matrix<float>(n_in, n_out);
    weight.randomize(-scalar, scalar);
    Wgrad = Matrix<float>(n_in, n_out); //defaults to 0

    if (use_bias)
    {
        bias = Matrix<float>(1, n_out);
        bias.randomize(-scalar, scalar);
        Bgrad = Matrix<float>(1, n_out); //defaults to 0
    }
}

Matrix<float> Linear::forward(const Matrix<float>& X)
{
    assert (X.size(1) == weight.size(0));

    // X*W + b
    Matrix<float> Out = matrix::matmul(X, weight);
    if (use_bias)
        Out.add(bias); 

    return Out;
}

Matrix<float>
Linear::backward(const Matrix<float>& X, const Matrix<float>& grad_flow)
{
    Matrix<float> dWeight, dBias, dInput;

    // Gradient w respect to weights:
    // Calc derivative & then add to gradient matrix
    auto xT = matrix::transpose(X);
    dWeight = matrix::matmul(xT, grad_flow);

    assert (dWeight.size(0) == Wgrad.size(0));
    assert (dWeight.size(1) == Wgrad.size(1));
    Wgrad.add(dWeight);

    // Gradient w respect to bias:
    if (use_bias)
    {
        dBias = grad_flow;
        dBias.colApply(matrix::vecSum); 

        assert (dBias.size(0) == Bgrad.size(0));
        assert (dBias.size(1) == Bgrad.size(1));
        Bgrad.add(dBias);
    }
    // Gradient w repsect to layer's inputs:
    auto wT = matrix::transpose(weight);
    dInput = matrix::matmul(grad_flow, wT);

    return dInput;
}




