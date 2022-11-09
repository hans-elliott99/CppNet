#include <iostream>
#include <vector>
#include <assert.h>
#include <functional>

#include "./functions.h"

#pragma once


template <typename T>
class Matrix
{
private:
    size_t _shape_i;
    size_t _shape_j;
    
public:
    std::vector<std::vector<T>> data;

public:
    Matrix(size_t shape_i = 1, size_t shape_j = 1);
    ~Matrix();

    // Modify inplace    
    void fill(T value);

    void identity(T value = 1);

    void randomize(int low = 0, int high = 1);

    void add(const Matrix& B);

    void apply(std::function<T(T)> fun);
    
    // Print info
    void print(int nrow = -1);

    void shape();

public:
    //operators
    Matrix operator+(const T& scalar);
    Matrix operator-(const T& scalar);
    Matrix operator*(const T& scalar);
    Matrix operator/(const T& scalar);
    std::vector<T>& operator[](int idx);

    template <typename Y>
    friend std::ostream& operator<<(std::ostream& stream, const Matrix<Y>& matrix);

    //vector methods
    size_t size(size_t axis = 0);
};


template <typename T>
Matrix<T> matmul(const Matrix<T>& A, const Matrix<T>& B);

template <typename T>
Matrix<T> addition(const Matrix<T>& A, const Matrix<T>& B);

// template <typename T> 
// Matrix<T*> viewT();
template <typename T>
Matrix<T> colApply(const Matrix<T>& A, T (*f)());
// colApply(Matrix<T>& A, std::function<T(T)> fun);

template <typename T> T vecSum(std::vector<T> vec);



// https://stackoverflow.com/questions/8752837/undefined-reference-to-template-class-constructor
template class Matrix<float>;
template class Matrix<double>;
template class Matrix<int>;