#include "optimizer.hpp"



void
Optim::add_layer(Layer & layer)
{
    _layers.push_back(&layer);
}

void
Optim::zero_grad()
// Set Gradients to Zero
{
    for (auto l : _layers)
    {
        l->Wgrad.zero();
        if (l->use_bias)
            l->Bgrad.zero();
    }
}

void
OptimSGD::step(float learning_rate)
{
    for (auto l : _layers)
    {
        l->weight.add(-learning_rate * l->Wgrad);
        if (l->use_bias)
            l->bias.add(-learning_rate * l->Bgrad);
    }
}