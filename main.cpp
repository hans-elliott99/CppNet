
#include "net/layer.hpp"
#include "data/dataset.hpp"
#include "utils/functions.hpp"
#include "matrix/matrix.tpp"
// #include "matrix/matrix.cpp" //have to include this since the linker can't find Matrix<type> just from the header 
                            //https://stackoverflow.com/questions/1639797/template-issue-causes-linker-error-c


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
    Xdev.shape();

    // MLP
    size_t neurons = 5;
    Matrix<float> fake_grad(Xdev.size(), neurons);
    fake_grad.randomize(0.001, 0.01);

    Linear l1(Xdev.size(1), neurons);
    

    auto output = l1.forward(Xdev);
    auto l1back = l1.backward(Xdev, fake_grad);
    l1back.shape();

    l1.Wgrad.print();
    std::cout << '\n';
    l1.Bgrad.print();
}


// int main()
// {
//     Matrix<float> A(5,5); A.fill(1);
//     Matrix<float> B(5,5); B.diagonal(2);
//     B(0,0) = 1; B(0,1) = 1; B(1,0) = 3;

//     A.print();
//     auto C = matrix::matmul(A, B);
//     std::cout << '\n';
//     C.print();

//     std::cout << '\n';
//     B.print();
// }