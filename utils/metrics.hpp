#include "../matrix/matrix.hpp"

namespace metrics
{


float accuracy(Matrix<float> pred, Matrix<float> truth)
{
    assert(pred.size(0)==truth.size(0) && "Accuracy: Number of samples do not match.");
    assert(pred.size(1)==truth.size(1) && "Accuracy: requires Nx1 matrices");

    int n_correct = 0;
    for (size_t i = 0; i < pred.size(0); i++)
    {
        if (pred.data[i] == truth.data[i])
            n_correct++;
    }

    return n_correct / static_cast<float>( pred.size(0) );
};


};