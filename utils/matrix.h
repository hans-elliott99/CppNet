#include <iostream>
#include <vector>
#include <assert.h>
#include "./functions.h"

#pragma once


class Matrix
{
    size_t _shape_i;
    size_t _shape_j;
    
public:
    std::vector<std::vector<double>> M;

    Matrix(size_t shape_i=1, size_t shape_j=1);

    void rowfill(size_t idx, const std::vector<double> &datarow);
    
    void randomize(int low = 0, int high = 1);

    void print(int nrow = -1);

    void shape();
};

