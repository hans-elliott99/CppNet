#include "net/loss.hpp"
#include "net/layer.hpp"
#include "net/functions.hpp"
#include "data/dataset.hpp"
#include "matrix/matrix.hpp"

// TODO : chainining operators -> clean up nn code. finish sigmoid, add loss fn

int main()
{
    // // Load & split data
    // Dataset data("C:\\Users\\hanse\\Documents\\ml_code\\CppNet\\data1000.txt");
    // // data.head(5);
    // data.make_split(0.8);

    // // Convert to Matrix and retrieve Xs & ys
    // auto Xtr = data.toMatrix(DataSplit::TRAIN);
    // auto Ytr = data.toMatrix(DataSplit::TRAIN, true);
    // auto Xdev = data.toMatrix(DataSplit::TEST);
    // auto Ydev = data.toMatrix(DataSplit::TEST, true);


    // // MLP
    // size_t neurons1 = 5;
    // size_t neurons2 = 6;
    // size_t out_neurons = 1;

    // // Structure
    // ReLU l1(X.size(1), neurons1);
    // Linear l2(neurons1, neurons2); 
    // Sigmoid l3(neurons2, out_neurons);
    
    // // Forward
    // auto l1_out = l1.forward(X);
    // auto l2_out = l2.forward(l1_out);
    // auto l3_out = l3.forward(l2_out);

    // l3_out.shape();

    // // Backward
    // /*pretend loss --> fake_grad*/
    // Matrix<float> fake_grad(X.size(0), out_neurons); fake_grad.randomize(0.001, 0.01);

    // Matrix<float> grad_flow;
    // grad_flow = l3.backward(l2_out, fake_grad);
    // grad_flow = l2.backward(l1_out, grad_flow);
    // grad_flow = l1.backward(X,      grad_flow);


    // l2.Wgrad.print();
    // std::cout << '\n';
    
    // l2.Bgrad.print();
    // std::cout << '\n';

}





// int main()
// {
//     Matrix<float> A(2,2);

//     A.randomize().print(-1, 5);
//     A.apply(logf).print(-1, 5);
// }



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

