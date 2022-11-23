#include "mlp.hpp"


MLP::MLP(size_t inputDim,
        std::initializer_list<size_t> hiddenDims,
        std::initializer_list<LayerType> layerTypes
        ) :
          n_layers   ( layerTypes.size() ),
          _inputDim  ( inputDim          ),
          _neurons   ( hiddenDims        ),
          _layerTypes( layerTypes        )
{
    // Checks
    assert (hiddenDims.size() == layerTypes.size() && 
        "MLP: Length of hiddenDims list does not match that of layerTypes list."
    );


    // Initialize Neural Net Architecture
    for (size_t i = 0; i < n_layers; i++)
    {
        // Determine Previous Layer Output Dimension:
        size_t neurons_prev;
        if (i > 0)
            neurons_prev = _neurons[i-1];
        else
            neurons_prev = _inputDim;

        // Iterate through provided layer types and initialize layers
        auto lt = _layerTypes[i];
        switch (lt) {
            case (LayerType::DenseLinear):
                layers.emplace_back( new Linear(neurons_prev, _neurons[i]) );
                break;

            case (LayerType::DenseReLU):
                layers.emplace_back( new ReLU(neurons_prev, _neurons[i]) );
                break;
            
            case (LayerType::DenseSigmoid):
                layers.emplace_back( new Sigmoid(neurons_prev, _neurons[i]) );
                break;
        }
    }
}

Matrix<float>
MLP::forward(Matrix<float> &X)
{
    _X = &X; //store pointer to inputs for backprop

    Matrix<float> out;
    for (size_t i = 0; i < n_layers; i++)
    {
        // For each layer, pass in the outputs of the prev layer
        // (or X for first layer). Return the final layer's output.
        if (i == 0)
            out = (*layers[0]).forward(X);
        else
            out = (*layers[i]).forward(out);

        outputs.push_back( out );
    }
    return out;
}

void MLP::backward(Matrix<float> &loss_grad)
{
    Matrix<float> grad;
    for (size_t i = n_layers; i--;)
    {
        // For the top layer, pass back the gradients from loss function
        // For middle layers, pass back gradients successively
        // For input layer, pass back gradient and the original 
        if (i == n_layers-1)
        {
            grad = (*layers[i]).backward(
                outputs[i-1], loss_grad
                );
        }

        else if (i > 0)
        {
            grad = (*layers[i]).backward(
                outputs[i-1], grad
            );
        }

        else
        {
            grad = (*layers[i]).backward(
                *_X, grad
            );
        }
    }
}
