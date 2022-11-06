#include <iostream>
#include <cmath>
#include <vector>
#include <algorithm>


#pragma once

double sigmoid(double x);
double sigmoid_derivative(double x);

double relu(double x);
double relu_derivative(double x);

double random(double low, double high);

double euclidean_distance(const std::vector<double>& vec1,  const std::vector<double>& vec2);