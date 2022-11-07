
#include <iostream>
#include <vector>
#include <assert.h>
#include <algorithm>

#include "functions.h"
#include "matrix.h"


Matrix::Matrix(size_t shape_i, size_t shape_j)
{

    // Matrix of 0s by default 
    std::vector<std::vector<double>> M{shape_i, std::vector<double>(shape_j, 0.)};

    _shape_i = shape_i;
    _shape_j = shape_j;

}


void Matrix::rowfill(size_t idx, const std::vector<double> &datarow)
{
    assert (datarow.size() == _shape_j);
    M[idx] = datarow;
}


void Matrix::randomize(int low, int high)
{
    for (size_t i {0}; i < M.size(); i++)
        for (size_t j {0}; j < M.size(); j++)
            {
                M[i][j] = random(low, high);
            }
}


void Matrix::print(int nrow)
{
    if (nrow == -1) 
        {nrow = M.size(); }
    for (size_t i {0}; i < nrow; i++)
    {
        for (double e: M[i])
            {std::cout << e << " \t";}
        std::cout<<'\n';
    }
}



void Matrix::shape()
{
    std::cout << "(" << _shape_i << ", " << _shape_j << ")\n";
}



std::vector<std::vector<double>> 
matmul(std::vector<std::vector<double>> &A, std::vector<std::vector<double>> &B)
{
    assert (A[0].size() == B.size());

    std::vector<std::vector<double>> output;
    std::vector<double> row;
    double product;

    for (size_t i {0}; i < A.size(); i++) //rows in a
    {
        for (size_t j {0}; j < B[0].size(); j++) //cols in b
        {
            product = 0;
            for (size_t v {0}; v < A[0].size(); v++) //elements in each row
                { product += A[i][v] * B[v][j]; }
            
            row.push_back(product);
        }        
        output.push_back(row);
        row.clear();
    }

    return output;
}



// int main()
// {
//     std::vector<std::vector<double>> A = {{1., 2., 3.},
//                                           {1., 2., 3.}};
//     std::vector<std::vector<double>> B = {{1., 2.}, 
//                                           {1., 2.}, 
//                                           {1., 2.}};
//     std::vector<std::vector<double>> C;

//     C = matmul(A, B);
//     // print_matrix(C);

//     Matrix m(5, 5);
//     m.randomize(-1, 1);
//     m.print();
// }