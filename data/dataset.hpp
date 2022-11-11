#include <string>
#include <fstream>
#include <algorithm>
#include <vector>
#include <iostream>
#include "../net/functions.hpp"
#include "../matrix/matrix.hpp"

#pragma once

enum DataSplit {TRAIN=0, TEST};

class Dataset
{
private:
       
    std::vector<std::vector<float>> _X;
    std::vector<std::vector<float>> _Y;
      
    std::vector<const std::vector<float>*> _Xtrain;
    std::vector<const std::vector<float>*> _Ytrain;
    std::vector<const std::vector<float>*> _Xtest;
    std::vector<const std::vector<float>*> _Ytest;

private:

    const std::vector<const std::vector<float>*>&
                    _getsplit(DataSplit s, bool label=false) const;

public:
    //Construct
    Dataset(std::string filename);
    //Destruct
    ~Dataset();    

    // Training, Testing Splits
    void make_split(double ptrain);

    // Dataset dimensions
    // void printshape(DataSplit s);
    std::vector<size_t> shape(DataSplit s, bool lab=false, bool print=false);

    // Print data
    void head(const int nrows);

    // convert to matrix
    // template <typename T>
    Matrix<float> toMatrix(DataSplit s, bool label=false);

    void test();

};