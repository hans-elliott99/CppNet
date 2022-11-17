
#include "./layer.hpp" 
#include "../matrix/matrix.hpp"

class Optim
{
protected:
    std::vector<Layer *> _layers;

public:
    void add_layer(Layer & layer);

    void zero_grad();

    virtual void step() {};
};


class OptimSGD : public Optim
{
public:
    void step(float learning_rate);
};