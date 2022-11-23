#include "utils/metrics.hpp"
#include "net/layer.hpp"
#include "net/loss.hpp"
#include "net/optimizer.hpp"
#include "net/mlp.hpp"


#include "net/functions.hpp"
#include "data/dataset.hpp"
#include "matrix/matrix.hpp"

// TODO: Saving & loading of weights
// TODO: Pass in dataset & specify model to a train.exe from command-line 
// TODO: Make an inference exe to predict given a saved model

std::vector<Matrix<float>>
load_dataset(double ptrain, bool shuffle)
{
    std::vector<Matrix<float>> data_list;

    // Load & split data
    Dataset data("C:\\Users\\hanse\\Documents\\ml_code\\CppNet\\data1000.txt");
    data.make_split(ptrain, shuffle);

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
    Matrix<float> x(10, 2); x.randomize();
    Matrix<float> y(10, 1); y.fill(1);
    BinaryCrossEntropy BCE;
    OptimSGD SGD;

    MLP mod(x.size(1),
            {32, 16, 1}, 
            {DenseLinear, DenseReLU, DenseSigmoid}
            );
    SGD.add_model(mod);

    Matrix<float> pred;
    for (size_t i; i < 100; i++)
    {
        SGD.zero_grad();
        pred = mod.forward(x);
        BCE.compute(pred, y);
        mod.backward(BCE.dInput);
        SGD.step(1.0);
        
        if (i % 5 == 0)
            std::cout << BCE.Loss << '\n';
    }

    pred.print();
    std::cout << mod.n_layers;

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

