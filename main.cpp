
#include "net/layer.hpp"
#include "net/functions.hpp"
#include "data/dataset.hpp"
#include "matrix/matrix.hpp"


int main()
{
    // Load & split data
    Dataset data("C:\\Users\\hanse\\Documents\\ml_code\\CppNet\\data1000.txt");
    // data.head(5);
    data.make_split(0.8);

    // Convert to Matrix and retrieve Xs & ys
    auto Xtr = data.toMatrix(DataSplit::TRAIN);
    auto Ytr = data.toMatrix(DataSplit::TRAIN, true);
    auto Xdev = data.toMatrix(DataSplit::TEST);
    auto Ydev = data.toMatrix(DataSplit::TEST, true);

    // MLP
    size_t neurons1 = 5;
    size_t out_neurons = 2;
    // fake gradient coming from nonexistant loss fn 
    Matrix<float> fake_grad(Xdev.size(), out_neurons);
    fake_grad.randomize(0.001, 0.01);

    // Structure
    ReLU l1(Xdev.size(1), neurons1);
    Linear l2(neurons1, out_neurons); 
    
    // Forward
    auto l1_out = l1.forward(Xdev);
    auto l2_out = l2.forward(l1_out);

    // Backward
    /*pretend loss --> fake_grad*/
    auto l2_back = l2.backward(l1_out, fake_grad);
    auto l1_back = l1.backward(Xdev, l2_back);




    l2.Wgrad.print();
    std::cout << '\n';
    
    l2.Bgrad.print();
    std::cout << '\n';

}



// TODO : finish sigmoid


// int main()
// {

//     Matrix<float> A(5,5); A.fill(3);
//     Matrix<float> B(5,5); B.diagonal(2);
//     // B(0,0) = 1; B(0,1) = 1; B(1,0) = 3;
//     auto out = 2.0f + A;
//     out.print();
// }