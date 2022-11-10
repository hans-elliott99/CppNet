
#include <iostream>
#include <vector>
#include <assert.h>
#include <algorithm>

#include "matrix.h"

// Default constructor
template <typename T>
Matrix<T>::Matrix(size_t shape_i, size_t shape_j):
        _shape_i(shape_i), 
        _shape_j(shape_j), 
        data(shape_i*shape_j, 0)
{}

/**
 * OPERATORS
*/
template <typename T> Matrix<T> 
Matrix<T>::operator+(const T& scalar)
{
    Matrix output(_shape_i, _shape_j);

    for (size_t i {0}; i < _shape_i; i++)
    {
        std::transform(
            data.begin(), data.end(),
            output.data.begin(),
            std::bind(std::plus<T>(), std::placeholders::_1, scalar)
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

    for (size_t i {0}; i < _shape_i; i++)
    {
        std::transform(
            data.begin(), data.end(),
            output.data.begin(),
            std::bind(std::multiplies<T>(), std::placeholders::_1, scalar)
        );
    }
    return output;
}

template <typename T> Matrix<T>
Matrix<T>::operator/(const T& scalar)
{
    return operator*(1 / scalar);
}

template <typename T> T& 
Matrix<T>::operator()(size_t i, size_t j)
{
    assert (i < _shape_i && "`i` index out of range.");
    assert (j < _shape_j && "`j` index out of range.");
    return data[i*_shape_i + j]; 
}

template <typename T> T const& 
Matrix<T>::operator()(size_t i, size_t j) const
{
    assert (i < _shape_i && "`i` index out of range.");
    assert (j < _shape_j && "`j` index out of range.");
    return data[i*_shape_i + j]; 
}


template <typename T> std::vector<T>
Matrix<T>::operator[](size_t i) 
{
    std::vector<T> slice(
        data.begin() + (i*_shape_j),
        data.begin() + (i*_shape_j + _shape_j)
    );
    return slice;
}


template <typename Y> std::ostream& 
operator<<(std::ostream& stream, Matrix<Y>& matrix)
{ 
    stream << '\n';
    for (size_t i {0}; i < matrix._shape_i; i++)
    {
        for (size_t j{0}; j < matrix._shape_j; j++)
            {std::cout << matrix.data[i*matrix._shape_j + j] << " \t";}
        std::cout<<'\n';
    }
    return stream;
}


/**
 * Access subvectors
*/
template <typename T> std::vector<T> 
Matrix<T>::Row(int row, int colBegin, int colEnd) const
{
    if (colEnd < 0) colEnd = _shape_j - 1;
    if (colBegin <= colEnd)
        return _stridedSlice(index(row, colBegin), colEnd-colBegin+1, 1);
    else
        return _stridedSlice(index(row, colBegin), colBegin-colEnd+1, -1);
}

template <typename T> std::vector<T> 
Matrix<T>::Column(int col, int rowBegin, int rowEnd) const
{
    if (rowEnd < 0) rowEnd = _shape_i - 1;
    if (rowBegin <= rowEnd)
        return _stridedSlice(index(rowBegin, col), rowEnd-rowBegin+1, _shape_j);
    else
        return _stridedSlice(index(rowBegin, col), rowBegin-rowEnd+1, -_shape_j);
}

// template <typename T> Matrix<T>
// Matrix<T>::copy() const
// {
//     Matrix<T> out(0,0);
//     out.data = data;
//     return out;
// }



template <typename T> std::vector<T>
Matrix<T>::_stridedSlice(int start, int length, int stride) const
{
    //https://stackoverflow.com/questions/15778377/get-the-first-column-of-a-matrix-represented-by-a-vector-of-vectors
    std::vector<T> out;
    out.reserve(length);
    
    const T* pos = &data[start];
    for (int i {0}; i < length; i++)
    {
        out.push_back(*pos);
        pos += stride; //increase value of pointer to move to next element in vector
    }
    return out;
}



/**
 * std::vector-like methods
*/
template <typename T> size_t 
Matrix<T>::size(size_t axis) const
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
    const size_t Brows = B.size(0);
    const size_t Bcols = B.size(1);

    assert (_shape_i == Brows); assert (_shape_j == Bcols);

    // Modify the matrix's data inplace
    std::transform(
        data.begin(), data.end(),
        B.data.begin(),
        data.begin(),
        std::plus<T>()
    );
}

template <typename T> void 
Matrix<T>::fill(T value)
{
    data = std::vector<T>(_shape_i*_shape_j, value);
}

template <typename T> void 
Matrix<T>::diagonal(T value)
{
    assert (_shape_i == _shape_j); //n x n only
    for (size_t i {0}; i < _shape_i; i++)
        data[i*_shape_i + i] = value;
}

template <typename T> void 
Matrix<T>::randomize(int low, int high)
{
    for (size_t i {0}; i < _shape_i; i++)
        for (size_t j {0}; j < _shape_j; j++)
        {
            data[i*_shape_i + j] = random<T>(low, high);
        }
}

/**
 * APPLY
*/

template <typename T> void 
Matrix<T>::apply(std::function<T(T)> fun)
{
    for (size_t i {0}; i < _shape_i; i++)
    {
        std::transform(
            data.begin(), data.end(),
            data.begin(),
            fun
        );
    }
}

template <typename T> void
Matrix<T>::colApply(T (*fun)(std::vector<T>&))
{
    std::vector<T> result;
    result.reserve(_shape_j);

    std::vector<T> col;
    T value;
    for (size_t j {0}; j < _shape_j; j++)
    {
        col = Column(j); //make fn that returns pointers instead of copying the column's data?
        value = (*fun)(col);   
        result.push_back(value);
    }

    data = result;
    _shape_i = 1;
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
        for (size_t j {0}; j < _shape_j; j++)
            {std::cout << data[i*_shape_i + j] << " \t";}
        std::cout<<'\n';
    }
}

template <typename T> void 
Matrix<T>::shape()
{
    std::cout << "(" << _shape_i << ", " << _shape_j << ")\n";
}




// // Functions
// template <typename T> Matrix<T> 
// matmul(const Matrix<T>& A, const Matrix<T>& B)
// {
//     const size_t Arows = A.data.size();
//     const size_t Acols = A.data[0].size();
//     const size_t Brows = B.data.size();
//     const size_t Bcols = B.data[0].size();
//     assert (Acols == Brows);

//     Matrix<T> output(Arows, Bcols);
//     std::vector<T> row;
//     T product;

//     for (size_t i {0}; i < Arows; i++) //rows in a
//     {
//         for (size_t j {0}; j < Bcols; j++) //cols in b
//         {
//             product = 0;
//             for (size_t v {0}; v < Acols; v++) //elements in each row
//                 { product += A.data[i][v] * B.data[v][j]; }
            
//             row.push_back(product);
//         }        
//         output.data[i] = {row};
//         row.clear();
//     }

//     return output;
// }

// template <typename T> Matrix<T> 
// addition(const Matrix<T>& A, const Matrix<T>& B)
// {
//     const size_t Arows = A.data.size();
//     const size_t Acols = A.data[0].size();
//     const size_t Brows = B.data.size();
//     const size_t Bcols = B.data[0].size();

//     assert (Arows == Brows); assert (Acols == Bcols);

//     Matrix output(Arows, Acols);
//     for (size_t i {0}; i < Arows; i++)
//     {
//         std::transform(A.data[i].begin(), A.data[i].end(), B.data[i].begin(),
//                        output.data[i].begin(),
//                        std::plus<T>()
//                        );
//     }
//     return output;
// }




template <typename T>
T vecSum(std::vector<T> &vec)
{
    T sum = 0;
    for (auto& n : vec)
        sum += n;
    return sum;
}


template <typename T>
T random(int low, int high)
{
    return low + static_cast<T>(rand()) / ( static_cast<T>(RAND_MAX / (high - low)) );
}