

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

    // error - if using type other than float, linker gets lost when it s
    // sees Dataset initializing a matrix of type float
    Matrix<float> A(5, 5); A.randomize();
    Matrix<float> B(5, 3); B.fill(-1.0);

    auto C = matmul(A, B);
    // C.apply(sigmoid);

    C.print();
}