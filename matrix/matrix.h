#include <iostream>
#include <iomanip>
#include <vector>
#include <assert.h>
#include <functional>
#include <algorithm>

#pragma once

#define FIXED_PRECIS(x) std::fixed <<std::setprecision(3)<<(x)

// Row-major ordering
// Element [i,j] is at (i * _shape_j + j)
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

    size_t nelements() {return data.size(); }
    
    size_t index(size_t i, size_t j ) const { return i * _shape_j + j; }

    // Modify inplace    
    ////implement chaining https://blog.stratifylabs.dev/device/2020-12-15-Method-Chaining-in-Cpp/#:~:text=Method%20chaining%20in%20C%2B%2B%20is,another%20method%20can%20be%20called.
    void fill(T value);
    void zero();
    
    void diagonal(T value = 1);

    void randomize(double low = 0, double high = 1);

    void add(const Matrix& B);

    void apply(std::function<T(T)> fun);

    void colApply(T (*fun)(std::vector<T>&));

    void rowApply(T (*fun)(std::vector<T>&));

    void transpose();

    // Access elements
    std::vector<T> Row( int row, int colBegin = 0, int colEnd = -1 ) const;

    std::vector<T> Column( int col, int rowBegin = 0, int rowEnd = -1 ) const;
    //// Matrix<T> copyslice(int rowBegin = 0, int colBegin = 0, int rowEnd = -1,  int colEnd = -1) const;

    // Broadcasting
    Matrix<T> broadcast(size_t dim, size_t length) const;
 
    // Print info
    void print(int nrow = -1);

    void shape();


public:
    //operators
    Matrix operator+(const T& scalar);
    Matrix operator-(const T& scalar);
    Matrix operator*(const T& scalar);
    Matrix operator/(const T& scalar);
    T& operator()(size_t i, size_t j);
    T const& operator()(size_t i, size_t j) const;
    std::vector<T> operator[](size_t i);

    template <typename Y>
    friend std::ostream& operator<<(std::ostream& stream, Matrix<Y>& matrix);

    //vector methods
    size_t size(size_t axis = 0) const;
};

namespace matrix
{
    template <typename T> Matrix<T> matmul(const Matrix<T>& A, const Matrix<T>& B);

    template <typename T> Matrix<T> matsum(const Matrix<T>& A, const Matrix<T>& B);

    template <typename T> Matrix<T> transpose(const Matrix<T>& A);

    template <typename T> 
    T random(double low, double high);

    template <typename T> T vecSum(std::vector<T> &vec);
}

// https://stackoverflow.com/questions/8752837/undefined-reference-to-template-class-constructor
template class Matrix<float>;
template class Matrix<double>;
template class Matrix<int>;