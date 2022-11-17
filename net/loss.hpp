#include <iostream>
#include <vector>
#include <math.h>
#include "../matrix/matrix.hpp"
#include "../matrix/matrix.tpp"

#pragma once


class BinaryCrossEntropy
{
public:
    float Loss;
    Matrix<float> dInput;

    void compute(const Matrix<float>& y_pred, const Matrix<float>& y_true);

private:
    void _forward(const Matrix<float>& y_pred, const Matrix<float>& y_true);
    void _backward(const Matrix<float>& y_true);
private:
    Matrix<float> clipped_preds;
    Matrix<float> _clip_values(const Matrix<float> &Mat, float epsilon = 1.0e-07F);
};


