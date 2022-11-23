#include "optimizer.hpp"



void
Optim::add_layer(Layer & layer)
{
    _layers.emplace_back( &layer );
}

void
Optim::add_model(MLP & model)
{
    for (size_t i=0; i < model.n_layers; i++)
        _layers.emplace_back( model.layers[i] );
}


void
Optim::zero_grad()
// Set Gradients to Zero
{
    for (size_t i = 0; i < _layers.size(); i++)
    {
        _layers[i]->Wgrad.zero();
        if (_layers[i]->use_bias)
            _layers[i]->Bgrad.zero();
    }
}

void
OptimSGD::step(float learning_rate)
{
    for (size_t i = 0; i < _layers.size(); i++)
    {
        _layers[i]->weight.add(-learning_rate * _layers[i]->Wgrad);
        if (_layers[i]->use_bias)
            _layers[i]->bias.add(-learning_rate * _layers[i]->Bgrad);
    }
}