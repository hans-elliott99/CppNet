# CppNet

This is a basic Neural Network built with the C++ standard library.  

First I built an oversimplified and probably inefficient matrix/linear algebra class (`matrix/matrix.hpp`) which is loosely inspired by python's numpy.  
Then I built some basic neural network layers with forward and backward methods, loosely inspired by pytorch (`net/layer.hpp`).  
I implemented binary crossentropy (`net/loss.hpp`) as a loss function and basic stochastic gradient descent (`net/optimizer.hpp`) to optimize the network.  

To compile this project with GNU g++ and Makefile:
> `git clone https://github.com/hans-elliott99/CppNet.git`   
> `cd ./CppNet`  
> `make`  
