#include "layer.hpp"
#include "../matrix/matrix.hpp"
#include <vector>
#include <utility>
#include <memory>
#include <assert.h>

#pragma once

class MLP
{

private:
    size_t _inputDim, _outputDim;
    std::vector<LayerType> _layerTypes;
    std::vector<size_t> _neurons;
    Matrix<float>* _X;



public:
    size_t n_layers = 0;
    std::vector< std::shared_ptr<Layer> > layers; //list of layers
    std::vector< Matrix<float> > outputs;         //list of layer outputs from one forward pass

public:
    MLP(size_t inputDim,
        std::initializer_list<size_t> hiddenDims,
        std::initializer_list<LayerType> layerTypes);
    
    MLP(size_t inputDim) : _inputDim(inputDim) {};

    void add_layer(LayerType layerType, size_t n_neurons);

    void initXavier(std::default_random_engine &gen); //TODO: add layer indexing

    Matrix<float> forward(Matrix<float> &X);

    void backward(Matrix<float> &loss_grad);
    
};