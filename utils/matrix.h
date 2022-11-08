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
    
    void fill(T value);

    void identity(T value = 1);

    void randomize(int low = 0, int high = 1);

    void add(const Matrix& B);

    void apply(std::function<T(T)> fun);

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