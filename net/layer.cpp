#include "layer.h"


Linear::Linear(size_t n_in, size_t n_out, bool use_bias) :
    weight(n_out, n_in),
    bias(n_out, 1),
    use_bias(use_bias)
{
    // Initialize
    double scalar = sqrt(static_cast<double>(1 / n_in));
    weight.randomize(-scalar, scalar);
    bias.randomize(-scalar, scalar);
}

Matrix<float> Linear::forward(Matrix<float> X)
{
    assert (X.size(0) == weight.size(1));

    // W*X + b
    Matrix<float> Out = matrix::matmul(weight, X);
    Out.add(bias); 

    return Out;
}




