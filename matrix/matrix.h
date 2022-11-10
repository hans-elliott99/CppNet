#include <iostream>
#include <vector>
#include <assert.h>
#include <functional>


#pragma once


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
    void fill(T value);

    void diagonal(T value = 1);

    void randomize(int low = 0, int high = 1);

    void add(const Matrix& B);

    void apply(std::function<T(T)> fun);

    // void colApply(std::function<T(std::vector<T>)> fun);
    void colApply(T (*fun)(std::vector<T>&));
    
    // Print info
    void print(int nrow = -1);

    void shape();

    // Access elements
    std::vector<T> Row( int row, int colBegin = 0, int colEnd = -1 ) const;
    std::vector<T> Column( int col, int rowBegin = 0, int rowEnd = -1 ) const;
    // Matrix<T> copy(int rowBegin = 0, int colBegin = 0, int rowEnd = -1,  int colEnd = -1) const;

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


template <typename T> Matrix<T> matmul(const Matrix<T>& A, const Matrix<T>& B);

template <typename T> Matrix<T> addition(const Matrix<T>& A, const Matrix<T>& B);

template <typename T> T random(int low, int high);

template <typename T> T vecSum(std::vector<T> &vec);



// https://stackoverflow.com/questions/8752837/undefined-reference-to-template-class-constructor
template class Matrix<float>;
template class Matrix<double>;
template class Matrix<int>;