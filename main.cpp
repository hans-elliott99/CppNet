

#include "data/dataset.h"
#include "utils/functions.h"
#include "utils/matrix.h"
#include "utils/matrix.cpp" //have to include this since the linker can't find Matrix<type> just from the header 
                            //https://stackoverflow.com/questions/1639797/template-issue-causes-linker-error-c

int main()
{
    // // Load & split data
    // Dataset data("C:\\Users\\hanse\\Documents\\ml_code\\CppNet\\data1000.txt");
    // // data.head(5);
    // data.make_split(0.8);

    // // Convert to Matrix and retrieve Xs & ys
    // auto Xtr = data.toMatrix(DataSplit::TRAIN);
    // auto Ytr = data.toMatrix(DataSplit::TEST, true);
    // Xtr.shape();
    // Ytr.shape();

    Matrix<float> A(5, 5); A.randomize(-1, 1);
    Matrix<float> B(3, 3); B.identity();

    // auto C = matmul(A, B);
    // C.apply(relu);
    // C.print();
    B.print();
    std::cout << '\n';

    // problem passing fn to colapply...
    // also accessing cols in vec of vec is slow compared to a flat vector...
    auto D = colApply(B, vecSum<float>);
    // B.apply(relu);
    D.print();
}