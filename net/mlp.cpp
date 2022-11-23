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
    // Initialize model and specify structure using list initializtion

    // Checks
    assert (hiddenDims.size() == layerTypes.size() && 
        "MLP: Length of hiddenDims list does not match that of layerTypes list."
    );

    // Initialize Neural Net Architecture
    layers.reserve( n_layers );
    _neurons.reserve( n_layers );

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

void
MLP::add_layer(LayerType layerType, size_t n_neurons)
{
    // Add a single layer to the model by specifying layer type and number of neurons

    // Determine number of neurons in previous layer
    size_t neurons_prev;

    if (n_layers == 0)
        neurons_prev = _inputDim;
    else
        neurons_prev = _neurons[_neurons.size()-1]; // the previous layer's neurons

    switch (layerType) {
        case (LayerType::DenseLinear):
            layers.emplace_back( new Linear(neurons_prev, n_neurons) );
            break;

        case (LayerType::DenseReLU):
            layers.emplace_back( new ReLU(neurons_prev, n_neurons) );
            break;
        
        case (LayerType::DenseSigmoid):
            layers.emplace_back( new Sigmoid(neurons_prev, n_neurons) );
            break;
    }

    // Save # of neurons and increment the # of layers
    _neurons.push_back( n_neurons );
    n_layers++;

}

void 
MLP::initXavier(std::default_random_engine &gen)
{
    for (size_t i = 0; i < n_layers; i++)
    {
        float gain;
        std::string name { (*layers[i]).name };
        if (name == "DenseLinear") gain = 1.0F;
        if (name == "DenseReLU") gain = 1.4142135623730951F;
        if (name == "DenseSigmoid") gain = 1.0F;

        (*layers[i]).initXavierNormal(gen, gain);
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
