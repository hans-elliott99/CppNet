#include "net/optimizer.hpp"
#include "net/loss.hpp"
#include "net/layer.hpp"
#include "net/functions.hpp"
#include "data/dataset.hpp"
#include "matrix/matrix.hpp"


std::vector<Matrix<float>>
load_dataset(double ptrain)
{
    std::vector<Matrix<float>> data_list;

    // Load & split data
    Dataset data("C:\\Users\\hanse\\Documents\\ml_code\\CppNet\\data1000.txt");
    data.make_split(ptrain);

    // Convert to Matrix and retrieve Xs & ys
    auto Xtr = data.toMatrix(DataSplit::TRAIN);
    auto Ytr = data.toMatrix(DataSplit::TRAIN, true);
    auto Xdev = data.toMatrix(DataSplit::TEST);
    auto Ydev = data.toMatrix(DataSplit::TEST, true);

    data_list.push_back(Xtr); data_list.push_back(Ytr);
    data_list.push_back(Xdev); data_list.push_back(Ydev);
    return data_list;
}


int main()
{
    Matrix<float> X, Y;
    auto data_list = load_dataset(0.8);
    X = data_list[0];
    Y = data_list[1];
    std::cout << "X.shape = "; X.shape();
    std::cout << "Y.shape = "; Y.shape();

    // Matrix<float> X(6,6); X.randomize(-1, 1);
    // Matrix<float> Y(6,1); Y.fill(1);

    // MLP
    size_t neurons1 = 128;
    size_t neurons2 = 64;
    size_t out_neurons = 1;

    // Structure
    ReLU l1(X.size(1), neurons1);
    ReLU l2(neurons1, neurons2); 
    Sigmoid l3(neurons2, out_neurons);

    // Optimizer
    OptimSGD sgd;
    sgd.add_layer(l1); sgd.add_layer(l2); sgd.add_layer(l3);

    // Training Loop
    Matrix<float> l1_out, l2_out, l3_out;
    BinaryCrossEntropy BCE;
    Matrix<float> grad_flow;


    for (size_t step = 0; step < 100; step++)
    {
        sgd.zero_grad();
        // Forward
        l1_out = l1.forward(X);
        l2_out = l2.forward(l1_out);
        l3_out = l3.forward(l2_out);

        // Loss
        BCE.compute(l3_out, Y);

        // Backward
        grad_flow = BCE.dInput;
        grad_flow = l3.backward(l2_out, grad_flow);
        grad_flow = l2.backward(l1_out, grad_flow);
        grad_flow = l1.backward(X,      grad_flow);

        // Optimize
        sgd.step(2);

        // Console Updates
        if (step % 5 == 0)
           std::cout << step << "\t| Loss = " << BCE.Loss << '\n'; 
    }

    l3_out.print(5);
    std::cout << "\n\n";
    Y.print(5);

    // l1.Wgrad.print();
    // std::cout << '\n';
    
    // l1.Bgrad.print();
    // std::cout << '\n';

}





// python:
// 
//     import torch
//     import numpy as np
//     X = torch.ones((5,1)) / 2; X.requires_grad = True
//     Y = torch.ones((5,1))
//     Xsig = torch.sigmoid(X); Xsig.retain_grad()
//     loss = torch.nn.functional.binary_cross_entropy(Xsig, Y)
//     print(loss)
//     print(Xsig.grad)

// C++:
// 
//     Matrix<float> X(5, 1); X.fill(0.5);
//     Matrix<float> Y(5, 1); Y.fill(1);
//     X.apply(sigmoid);     
//     BinaryCrossEntropy L;
//     L.compute(X, Y);
//     std::cout << "LOSS: " <<  L.Loss << '\n';
//     L.dInput.print();

