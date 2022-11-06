#include "dataset.h"
#include "../utils/functions.h"


Dataset::Dataset(std::string filename)
{
    std::ifstream in_file(filename);
    if (in_file.fail()) 
        {std::cout << "File not found." << '\n'; }

    std::vector<double> row;
    double a;
    while (in_file >> a)
    {
        row.push_back(a);

        if (in_file.get() == '\n')
        {
            double label {row[row.size() - 1]};
            row.pop_back(); //remove last element in vector (ie, the newline)

            // Add samples and labels to data vectors
            _X.push_back(row); //add new element at end of vector
            _Y.push_back({label});
            
            row.clear(); //reset row 
        }
    }         
}
Dataset::~Dataset() {}

// Training, Testing Splits
void Dataset::make_split(double ptrain)
{
    for (size_t i {0}; i < _X.size(); i++)
    {
        if (random(0, 1) < ptrain) //functions.h
        {
            // Sample goes to training split
            _Xtrain.push_back(&_X[i]) ;
            _Ytrain.push_back(&_Y[i]);
        }
        else
        {
            // Samples goes to testing split
            _Xtest.push_back(&_X[i]) ;
            _Ytest.push_back(&_Y[i]);
        }
    }
}

void Dataset::shape(DataSplit s)
{
    std::vector<const std::vector<double>*> data;
    switch(s)
    {
    case DataSplit::TRAIN:
        data = _Xtrain;
        break;
    case DataSplit::TEST:
        data = _Xtest;
        break;
    }

    std::vector<unsigned long long> shape;
    shape = {data.size(), data.size()};
    
    std::cout << '(';
    for (size_t i {0}; i < shape.size(); i++) //scalable to more than 2 dims
        {
            std::cout << shape[i] << ", ";
        }        
        std::cout << "\b\b)";
}

void Dataset::head(const int nrows)
{
    for (size_t i {0}; i < nrows; i++)
    {
        for (double e: _X[i])
            {std::cout << "| " << e << '\t';}
        for (double e: _Y[i]) 
            {std::cout << "| label: " << e << '\n';}
            
    }
}

const std::vector<const std::vector<double>*>&
Dataset::inputs(DataSplit s) const                       //https://stackoverflow.com/questions/3141087/what-is-meant-with-const-at-end-of-function-declaration
{
    if (DataSplit::TRAIN)
        return _Xtrain;
    else
        return _Xtest;
}
    
const std::vector<const std::vector<double>*>&
Dataset::labels(DataSplit s) const
{
    if (DataSplit::TRAIN)
        return _Ytrain;
    else
        return _Ytest;
}



// int main()
// {
//     Dataset data("C:\\Users\\hanse\\Documents\\ml_code\\CppNet\\data1000.txt");

//     data.head(5);
//     data.make_split(0.8);

//     data.shape(DataSplit::TRAIN);
//     data.shape(DataSplit::TEST);

//     data.inputs(DataSplit::TRAIN);
//     data.inputs(DataSplit::TEST);
//     data.labels(DataSplit::TRAIN);
//     data.labels(DataSplit::TEST);

// }