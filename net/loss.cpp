#include "loss.hpp"


void
BinaryCrossEntropy::compute(const Matrix<float>& y_pred, const Matrix<float>& y_true)
{
    dInput.data.clear();

    _forward(y_pred, y_true);
    _backward(y_true);
}


void 
BinaryCrossEntropy::_forward(const Matrix<float>& y_pred, const Matrix<float>& y_true)
{
    Matrix<float> y_pred_clipped;

    // Clip preds (symmetrically) to avoid log of zero
    clipped_preds = _clip_values(y_pred);
    y_pred_clipped = clipped_preds; //keeping clipped preds untouched

    // Sample-wise loss
    //-( y_true * log(y_pred)) + (1-y_true)*log(1-y_pred) )
    Matrix<float> sample_loss;
    Matrix<float> log_ypc = y_pred_clipped; log_ypc.apply(logf);    //log(y_pred_clipped)
    Matrix<float> log_omypc = (1.0F - y_pred_clipped).apply(logf); //log(1 - y_pred_clipped)

    sample_loss = -1.0F * (
        log_ypc.mul(y_true)
    ).add(
        (1.0F - y_true).mul(log_omypc)
    );

    // Compute final loss and store
    sample_loss.rowApply(matrix::vecMean);
    Loss = matrix::mean(sample_loss);
}


void
BinaryCrossEntropy::_backward(const Matrix<float>& y_true)
{
    // Use clipped data to prevent division by zero
    Matrix<float> y_pred_clipped = clipped_preds;

    // Calculate gradient
    Matrix<float> dInputA, dInputB;
    const float samples = static_cast<float>(y_pred_clipped.size(0));
    const float outputs = static_cast<float>(y_pred_clipped.size(1));

    // (-y_true/y_pred + (1-y_true)/(1-y_pred)) / outputs
    // scaled by number of samples
    dInput = (
       -1.0F * matrix::divide(y_true, y_pred_clipped)
    ).add( 
       matrix::divide(1.0F - y_true, 1.0F - y_pred_clipped) 
    );
    dInput = (dInput / outputs) / samples; 
}

Matrix<float>
BinaryCrossEntropy::_clip_values(const Matrix<float>& Mat, float epsilon)
{
    Matrix<float> Mat_clipped = Mat;

    for (size_t i = 0; i < Mat_clipped.nelements(); i++)
    {
        if (Mat_clipped.data[i] >= (1.0F - epsilon))
        {
            Mat_clipped.data[i] = Mat_clipped.data[i] - epsilon;
        }
        else if (Mat_clipped.data[i] <= epsilon)
        {
            Mat_clipped.data[i] = epsilon;
        } 
        else {}
    }
    return Mat_clipped;
}