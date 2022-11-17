#include <iomanip>
#include <iostream>
#include <vector>
#include <assert.h>
#include <functional>
#include <algorithm>
#include <numeric>

#pragma once

#define ASSERT(condition) \
     { if(!(condition)){ std::cerr << "ASSERT FAILED: " << #condition << " @ " << __FILE__ << " (" << __LINE__ << ")" << std::endl; } }

// Row-major ordering: Element [i,j] is at (i * _shape_j + j)
template <typename T>
class Matrix
{
private:
    size_t _shape_i = 0;
    size_t _shape_j = 0;

    std::vector<T> _stridedSlice( int start, int length, int stride ) const;
    
public:
    typedef T value_type;
    
    std::vector<T> data;

public:
    // Constructor
    Matrix(size_t shape_i = 0, size_t shape_j = 0);
    ~Matrix() {};

    size_t nelements() const {return _shape_i*_shape_j; }
    
    size_t index(size_t i, size_t j ) const { return i * _shape_j + j; }

    std::vector<size_t> whichTrue(bool (*fun)(T)) const;

    // Modify inplace    
    ////implement chaining https://blog.stratifylabs.dev/device/2020-12-15-Method-Chaining-in-Cpp/#:~:text=Method%20chaining%20in%20C%2B%2B%20is,another%20method%20can%20be%20called.
    Matrix& fill(T value);
    Matrix& zero();
    
    Matrix& diagonal(T value = 1);

    Matrix& randomize(double low = 0, double high = 1);

    Matrix& add(const Matrix& B);

    Matrix& mul(const Matrix& B);

    Matrix& apply(std::function<T(T)> fun);

    Matrix& colApply(T (*fun)(std::vector<T>&));

    Matrix& rowApply(T (*fun)(std::vector<T>&));

    Matrix& transpose();

    // Access elements
    std::vector<T> Row( int row, int colBegin = 0, int colEnd = -1 ) const;
    std::vector<T>& viewRow( int row, int colBegin = 0, int colEnd = -1 ) const;

    std::vector<T> Column( int col, int rowBegin = 0, int rowEnd = -1 ) const;
        // Matrix copyslice(int rowBegin = 0, int colBegin = 0, int rowEnd = -1,  int colEnd = -1) const;

    // Broadcasting
    Matrix broadcast(size_t dim, size_t length) const;
 
    // Get info
    void print(int nrow = -1, int precis = 3, bool sci = false);

    void shape();
    
    double mean();
    double sd();
    T sum();

    //vector methods
    size_t size(size_t axis = 0) const;

    //operators
    T& operator()(size_t i, size_t j);

    T const& operator()(size_t i, size_t j) const;

    std::vector<T> operator[](size_t i);

    template <typename Y> friend std::ostream& operator<<(std::ostream& stream, Matrix<Y>& matrix);
};

namespace matrix
{
    template <typename T> Matrix<T> matmul(const Matrix<T>& A, const Matrix<T>& B);


    template <typename T> Matrix<T> matsum(const Matrix<T>& A, const Matrix<T>& B);

    template <typename T> Matrix<T> divide(const Matrix<T>& A, const Matrix<T>& B);

    template <typename T> Matrix<T> transpose(const Matrix<T>& A);

    template <typename T> T random(double low, double high);

    template <typename T> T vecSum(std::vector<T> &vec);

    template <typename T> T vecMean(std::vector<T> &vec);

    template <typename T> T mean(Matrix<T> &A);
}



/** OVERLOADED OPERATORS
 *  
 * Matrix & scalar combinations:
*/

// A + s
template <typename T> Matrix<T> 
operator+(const Matrix<T>& A, T scalar)
{
    Matrix<T> output(A.size(0), A.size(1));

    std::transform(
        A.data.begin(), A.data.end(),
        output.data.begin(),
        std::bind(std::plus<T>(), std::placeholders::_1, scalar)
    );
    return output;
}

// s + A
template <typename T> Matrix<T> 
operator+(T scalar, const Matrix<T>& A)
{ return A + scalar; }

// A - s
template <typename T> Matrix<T>
operator-(const Matrix<T>& A, T scalar)
{ return A + (-scalar); }

// s - A
template <typename T> Matrix<T>
operator-(T scalar, const Matrix<T>& A)
{
    Matrix<T> output(A.size(0), A.size(1));

    std::transform(
        A.data.begin(), A.data.end(),
        output.data.begin(),
        std::bind(std::minus<T>(), scalar, std::placeholders::_1)
    );
    return output;
}

// A * s
template <typename T> Matrix<T>
operator*(const Matrix<T>& A, T scalar)
{
    Matrix<T> output(A.size(0), A.size(1));

    std::transform(
        A.data.begin(), A.data.end(),
        output.data.begin(),
        std::bind(std::multiplies<T>(), std::placeholders::_1, scalar)
    );
    return output;
}

// s * A
template <typename T> Matrix<T>
operator*(T scalar, const Matrix<T>& A)
{return A * scalar; }

// A divide s
template <typename T> Matrix<T>
operator/(const Matrix<T>& A, T scalar)
{return A * (1 / scalar); }

// s divide A
template <typename T> Matrix<T>
operator/(T scalar, const Matrix<T>& A)
{
    Matrix<T> output(A.size(0), A.size(1));

    std::transform(
        A.data.begin(), A.data.end(),
        output.data.begin(),
        std::bind(std::divides<T>(), scalar, std::placeholders::_1) // check (does scalar get divided by element?)
    );
    return output;
}



// https://stackoverflow.com/questions/8752837/undefined-reference-to-template-class-constructor
template class Matrix<float>;
template class Matrix<double>;
template class Matrix<int>;