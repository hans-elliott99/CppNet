#include <string>
#include <fstream>
#include <algorithm>
#include <vector>
#include <iostream>

#include "../utils/functions.h"
#include "../utils/matrix.h"

#pragma once

enum DataSplit {TRAIN=0, TEST};

class Dataset
{
private:
       
    std::vector<std::vector<double>> _X;
    std::vector<std::vector<double>> _Y;
      
    std::vector<const std::vector<double>*> _Xtrain;
    std::vector<const std::vector<double>*> _Ytrain;
    std::vector<const std::vector<double>*> _Xtest;
    std::vector<const std::vector<double>*> _Ytest;



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

    // get inputs
    // const std::vector<const std::vector<double>*>& inputs(DataSplit s) const;
    // get labels
    // const std::vector<const std::vector<double>*>& labels(DataSplit s) const;

    // convert to matrix
    Matrix toMatrix();

};