

// #include "data/dataset.h"
#include "utils/functions.h"
#include "matrix/matrix.h"
#include "matrix/matrix.cpp" //have to include this since the linker can't find Matrix<type> just from the header 
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

    // Matrix<float> A(5, 5); A.randomize(-1, 1);
    // Matrix<float> B(3, 3); B.identity();

    // auto C = matmul(A, B);
    // C.apply(relu);
    // // C.print();
    // B.print();
    // std::cout << '\n';

    // problem passing fn to colapply...
    // also accessing cols in vec of vec is slow compared to a flat vector...
    // auto D = colApply(B, vecSum<float>);
    // B.apply(relu);
    // D.print();

    Matrix<float> A(1, 3);
    Matrix<float> B(3, 3); 
    
    // A(0,0) = 1; A(0,1) = 2; A(0,2) = 3;
    // A(0,0) = 1; A(1,0) = 2; A(2,0) = 3;
    // A.print();
    B(0,0) = 1; B(0,1) = 2; B(0,2) = 3;
    B(1,0) = 4; B(1,1) = 5; B(1,2) = 7;
    B(2,0) = 7; B(2,1) = 8; B(2,2) = 9;
    B.print();


    std::cout <<'\n';    

}