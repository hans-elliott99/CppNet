#include "layer.hpp"


Layer::Layer(size_t n_in, size_t n_out, bool use_bias) :
    use_bias(use_bias), 
    n_in(n_in), 
    n_out(n_out)
{
    name = "BaseLayer";
    // use_bias = use_bias;
    // n_in = n_in; n_out = n_out;

    weight = Matrix<float>(n_in, n_out);
    Wgrad = Matrix<float>(n_in, n_out); //defaults to 0

    if (use_bias)
    {
        bias = Matrix<float>(1, n_out);
        Bgrad = Matrix<float>(1, n_out); //defaults to 0
    }
}

Matrix<float> 
Layer::_layer_forward(const Matrix<float>& X)
{
    assert (X.size(1) == weight.size(0));

    // X*W + b
    Matrix<float> Out {matrix::matmul(X, weight)};
    if (use_bias)
        Out.add(bias); 

    return Out;
}

Matrix<float>
Layer::_layer_backward(const Matrix<float>& X, const Matrix<float>& grad_flow)
{
    Matrix<float> dWeight, dBias, dInput;

    // Gradient w respect to weights:
    // Calc derivative & then add to gradient matrix
    Matrix<float> xT {matrix::transpose(X)};
    dWeight = matrix::matmul(xT, grad_flow);

    assert (dWeight.size(0) == Wgrad.size(0));
    assert (dWeight.size(1) == Wgrad.size(1));
    Wgrad.add(dWeight);

    // Gradient w respect to bias:
    if (use_bias)
    {
        dBias = grad_flow;
        dBias.colApply(matrix::vecSum); 

        assert (dBias.size(0) == Bgrad.size(0)); 
        assert (dBias.size(1) == Bgrad.size(1));
        Bgrad.add(dBias);
    }
    // Gradient w repsect to layer's inputs:
    Matrix<float> wT {matrix::transpose(weight)};
    dInput = matrix::matmul(grad_flow, wT);

    return dInput;
}

void
Layer::_xavier_normal( float gain, std::default_random_engine &gen)
{
    float std2;
    float _in = static_cast<float>(n_in);
    float _out = static_cast<float>(n_out);

    // Weights come from Normal Dist (0, std^2) where
    // std = gain * sqrt(2 / (n_in + n_out))
    std2 = powf(
        (gain * sqrtf(2.0F / (_in + _out))) 
        ,2.0F
    );

    std::normal_distribution<float> norm(0, std2);

    // Populate weight data with 
    for (float& e : weight.data)
    {
        e = norm(gen);
    }
    if (use_bias) 
    {
        for (float& e : bias.data)
        {
            e = norm(gen);
        }
    }
}


////////////////////////////////////////////////////////
                    /*LINEAR*/
Linear::Linear(size_t n_in, size_t n_out, bool use_bias) :
    Layer(n_in, n_out, use_bias)
{
    name = "DenseLinear";
    // Initialize Weights
    double scalar {1.0 / n_in};
           scalar = sqrt(scalar);

    weight.randomize(-scalar, scalar);

    if (use_bias)
        bias.randomize(-scalar, scalar);
}

Matrix<float>
Linear::forward(const Matrix<float>& X)
{
    return _layer_forward(X); //no activ. fn applied
}

Matrix<float>
Linear::backward(const Matrix<float>& X, const Matrix<float>& grad_flow)
{
    return _layer_backward(X, grad_flow); //no activ. fn to backprop through
}


////////////////////////////////////////////////////////
                    /*RELU*/
ReLU::ReLU(size_t n_in, size_t n_out, bool use_bias) :
    Layer(n_in, n_out, use_bias)
{
    name = "DenseReLU";
    // Initialize Weights
    double scalar {1.0 / n_in};
           scalar = sqrt(scalar);

    weight.randomize(-scalar, scalar);

    if (use_bias)
        bias.randomize(-scalar, scalar);
}

Matrix<float>
ReLU::forward(const Matrix<float>& X)
{
    assert (X.size(1) == weight.size(0));

    _activ_inputs = _layer_forward(X);
    Matrix<float> Out {_activ_inputs}; //save activ inputs for backprop
    Out.apply(relu);
    
    return Out;
}

Matrix<float>
ReLU::backward(const Matrix<float>& X, const Matrix<float>& grad_flow)
{    
    /* Backprop the ReLU Activation */
    // (Where inp > 0, der is 1, else 0. Then we multiply by grad_flow
    // for chain rule, so can simplify to setting grad to 0 where inp < 0)
    Matrix<float> dReluInp {grad_flow};
    std::vector<size_t> zero_idx;
    auto is_below_zero = [](float e) {return (e <= 0.0F); };

    zero_idx = _activ_inputs.whichTrue(is_below_zero);
    
    for (size_t n : zero_idx)
    {
        // Set dReluInp to 0 at the "below zero" indices.
        dReluInp.data[n] = 0.;
    }
    _activ_inputs.data.clear(); //free up

    /* Backprop the Layer (passing back relu gradient) */
    Matrix<float> dInput {_layer_backward(X, dReluInp)};

    return dInput;
}

////////////////////////////////////////////////////////
                    /*SIGMOID*/

Sigmoid::Sigmoid(size_t n_in, size_t n_out, bool use_bias) :
    Layer(n_in, n_out, use_bias)
{
    name = "DenseSigmoid";
    // Initialize weights
    double scalar {1.0 / n_in};
        scalar = sqrt(scalar);

    weight.randomize(-scalar, scalar);

    if (use_bias)
        bias.randomize(-scalar, scalar);
}


Matrix<float>
Sigmoid::forward(const Matrix<float>& X)
{
    assert (X.size(1) == weight.size(0));

    _activ_outputs = _layer_forward(X); //save activ outputs for backprop
    _activ_outputs.apply(sigmoid);
    
    return _activ_outputs;
}

Matrix<float>
Sigmoid::backward(const Matrix<float>& X, const Matrix<float>& grad_flow)
{
    Matrix<float> dSigmoid;

    // grad_flow * (1 - sigmoid(x)) * sigmoid(x)
    dSigmoid = (1.0F - _activ_outputs).mul(_activ_outputs);
    dSigmoid.mul(grad_flow);

    // Backprop the layer
    Matrix<float> dInput {_layer_backward(X, dSigmoid)};

    return dInput;
}