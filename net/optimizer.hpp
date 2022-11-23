
#include "./layer.hpp" 
#include "./mlp.hpp"
#include "../matrix/matrix.hpp"

class Optim
{
protected:
    std::vector< std::shared_ptr<Layer> > _layers;

public:
    void add_layer(Layer & layer);

    void add_model(MLP & model);

    void zero_grad();

    virtual void step() {};
};


class OptimSGD : public Optim
{
public:
    void step(float learning_rate);
};