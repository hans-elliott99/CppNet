
#include <iostream>
#include <vector>
#include <assert.h>
#include <algorithm>

#include "functions.h"
#include "matrix.h"

template <typename T>
Matrix<T>::Matrix(size_t shape_i, size_t shape_j)
    : _shape_i(shape_i), _shape_j(shape_j)
{
    // Matrix of 0s by default 
    for (size_t i {0}; i < _shape_i; i++)
    {
        data.push_back(std::vector<T>(_shape_j, 0));
    }

    // _shape_i = shape_i;
    // _shape_j = shape_j;
}
template <typename T>
Matrix<T>::~Matrix() { }

/**
 * OPERATORS
*/
template <typename T> Matrix<T> 
Matrix<T>::operator+(const T& scalar)
{
    Matrix output(_shape_i, _shape_j);
    // Broadcast scalar into a j-length vector
    std::vector<T> vec(_shape_j, scalar);  
    for (size_t i {0}; i < _shape_i; i++)
    {
        std::transform(
            data[i].begin(), data[i].end(),
            vec.begin(),
            output.data[i].begin(),
            std::plus<T>()
        );
    }
    return output;
}

template <typename T> Matrix<T>
Matrix<T>::operator-(const T& scalar)
{
    return operator+(-scalar);
}

template <typename T> Matrix<T> 
Matrix<T>::operator*(const T& scalar)
{
    Matrix output(_shape_i, _shape_j);
    // Broadcast scalar into a j-length vector
    std::vector<T> vec(_shape_j, scalar);  
    for (size_t i {0}; i < _shape_i; i++)
    {
        std::transform(
            data[i].begin(), data[i].end(),
            vec.begin(),
            output.data[i].begin(),
            std::multiplies<T>()
        );
    }
    return output;
}

template <typename T> Matrix<T>
Matrix<T>::operator/(const T& scalar)
{
    return operator*(1 / scalar);
}

// Returns reference to the vector at the given index.
template <typename T> std::vector<T>& 
Matrix<T>::operator[](int idx)
{
    int nrows = static_cast<int>(_shape_i);
    if (abs(idx) > nrows)
        throw std::out_of_range("Index out of range in operator[].");
    if (idx < 0)
        idx = nrows - abs(idx);
    return data[idx];
}

template <typename Y> std::ostream& 
operator<<(std::ostream& stream, const Matrix<Y>& matrix)
{ 
    stream << '\n';
    for (size_t i {0}; i < matrix._shape_i; i++)
    {
        for (Y e: matrix.data[i])
            {std::cout << e << " \t";}
        std::cout<<'\n';
    }
    return stream;
}


/**
 * std::vector-like methods
*/
template <typename T> size_t 
Matrix<T>::size(size_t axis)
{
    if (axis == 0)
        return _shape_i;
    else
        return _shape_j;
}


/**
 * Modifiers
*/
template <typename T> void 
Matrix<T>::add(const Matrix<T>& B)
{
    const size_t Brows = B.data.size();
    const size_t Bcols = B.data[0].size();

    assert (_shape_i == Brows); assert (_shape_j == Bcols);

    for (size_t i {0}; i < _shape_i; i++)
    {
        // Modify the matrix's data inplace
        std::transform(
            data[i].begin(), data[i].end(),
            B.data[i].begin(),
            data[i].begin(),
            std::plus<T>()
        );
    }
}

template <typename T> void 
Matrix<T>::fill(T value)
{
    for (size_t i {0}; i < _shape_i; i++)
    {
        data[i] = std::vector<T>(_shape_j, value);
    }
}

template <typename T> void 
Matrix<T>::identity(T value)
{
    assert (_shape_i == _shape_j); //n x n only
    for (size_t e {0}; e < _shape_i; e++)
        data[e][e] = value;
}

template <typename T> void 
Matrix<T>::randomize(int low, int high)
{
    for (size_t i {0}; i < _shape_i; i++)
        for (size_t j {0}; j < _shape_j; j++)
        {
            data[i][j] = random(low, high);
        }
}


template <typename T> void 
Matrix<T>::apply(std::function<T(T)> fun)
{
    for (size_t i {0}; i < _shape_i; i++)
    {
        std::transform(
            data[i].begin(), data[i].end(),
            data[i].begin(),
            fun
        );
    }
}




/**
 * Utilities
*/
template <typename T> void 
Matrix<T>::print(int nrow)
{
    if (nrow == -1) 
        {nrow = _shape_i; }

    for (size_t i {0}; i < nrow; i++)
    {
        for (T e: data[i])
            {std::cout << e << " \t";}
        std::cout<<'\n';
    }
}

template <typename T> void 
Matrix<T>::shape()
{
    std::cout << "(" << _shape_i << ", " << _shape_j << ")\n";
}




// Functions
template <typename T> Matrix<T> 
matmul(const Matrix<T>& A, const Matrix<T>& B)
{
    const size_t Arows = A.data.size();
    const size_t Acols = A.data[0].size();
    const size_t Brows = B.data.size();
    const size_t Bcols = B.data[0].size();
    assert (Acols == Brows);

    Matrix<T> output(Arows, Bcols);
    std::vector<T> row;
    T product;

    for (size_t i {0}; i < Arows; i++) //rows in a
    {
        for (size_t j {0}; j < Bcols; j++) //cols in b
        {
            product = 0;
            for (size_t v {0}; v < Acols; v++) //elements in each row
                { product += A.data[i][v] * B.data[v][j]; }
            
            row.push_back(product);
        }        
        output.data[i] = {row};
        row.clear();
    }

    return output;
}

template <typename T> Matrix<T> 
addition(const Matrix<T>& A, const Matrix<T>& B)
{
    const size_t Arows = A.data.size();
    const size_t Acols = A.data[0].size();
    const size_t Brows = B.data.size();
    const size_t Bcols = B.data[0].size();

    assert (Arows == Brows); assert (Acols == Bcols);

    Matrix output(Arows, Acols);
    for (size_t i {0}; i < Arows; i++)
    {
        std::transform(A.data[i].begin(), A.data[i].end(), B.data[i].begin(),
                       output.data[i].begin(),
                       std::plus<T>()
                       );
    }
    return output;
}


// template <typename T> Matrix<const T*> 
// viewT(const Matrix<T>& A)
// {
//     size_t shape_i {A.data.size()};
//     size_t shape_j {A.data[0].size()};

//     Matrix<const T*> out(shape_j, shape_i);
    
//     for (size_t i {0}; i < shape_i; i++)
//         for (size_t j {0}; j <shape_j; j++)
//             out.data[i][j] = &(A.data[j][i]);

//     return out;
// }


template <typename T> Matrix<T> 
colApply(const Matrix<T>& A, T (*f)())
{
    size_t shape_i = A.size(0);
    size_t shape_j = A.size(1);

    Matrix<T> out(shape_j, 1);
    std::vector<T> column;

    for (size_t j {0}; j < shape_j; j++)
    {
        for (size_t i {0}; i < shape_i; i++)
        {
            column.push_back(A.data[i][j]);
        }
        out[j] = std::vector<T>(1, f(column));
        column.clear();
    }

    return out;
}


template <typename T> T
vecSum(std::vector<T>& vec)
{
    T sum = 0;
    for (auto& n : vec)
        sum += n;
}