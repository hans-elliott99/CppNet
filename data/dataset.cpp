#include "dataset.hpp"

Dataset::Dataset(std::string filename)
{
    std::ifstream in_file(filename);
    if (in_file.fail()) 
        {std::cout << "File not found." << '\n'; }

    std::vector<float> row;
    float a;
    while (in_file >> a)
    {
        row.push_back(a);

        if (in_file.get() == '\n')
        {
            float label {row[row.size() - 1]};
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
void Dataset::make_split(double ptrain, bool shuffle)
{   
    // Vector of row indices which can be randomly shuffle
    std::vector<size_t> rowIndx(_X.size());
    std::iota(rowIndx.begin(), rowIndx.end(), 0); //fill with 0...N
    if (shuffle)
    {
        // auto rd = std::random_device {}; //If we want to seed the shuffler
        auto rng = std::default_random_engine {}; //include rd() if seeding
        std::shuffle(std::begin(rowIndx), std::end(rowIndx), rng);
    }

    for (size_t i : rowIndx)
    {
        if (random<double>(0, 1) < ptrain) //functions.h
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


void Dataset::head(const int nrows)
{
    for (size_t i {0}; i < nrows; i++)
    {
        for (float e: _X[i])
            {std::cout << "| " << e << '\t';}
        for (float e: _Y[i]) 
            {std::cout << "| label: " << e << '\n';}
            
    }
}

Matrix<float> Dataset::toMatrix(DataSplit s, bool label)
{
    auto _d = _getsplit(s, label); 

    const size_t rows {_d.size()};
    const size_t cols {(*_d[0]).size()};
    Matrix<float> mat(rows, cols);

    for (size_t i {0}; i < rows; i++)
        for (size_t j {0}; j < cols; j++)
        {
            // mat.data[i*cols + j] = (*_d[i])[j];
            mat(i,j) = (*_d[i])[j];
        }
    
    return mat;
}

std::vector<size_t> 
Dataset::shape(DataSplit s, bool lab, bool print)
{
    std::vector<const std::vector<float>*> data;

    switch(s)
    {
    case DataSplit::TRAIN:
        if (!lab)
            data = _Xtrain;
        else 
            data = _Ytrain;
        break;
    case DataSplit::TEST:
        if (!lab)
            data = _Xtest;
        else
            data = _Ytest;
        break;
    }

    std::vector<size_t> shape {data.size(), (*data[0]).size()};

    if (print)
    {
        std::cout << '(';
        for (size_t e : shape) //scalable to more than 2 dims
            {
                std::cout << e << ", ";
            }        
            std::cout << "\b\b)\n";
    }

    return shape;
}


const std::vector<const std::vector<float>*>&
Dataset::_getsplit(DataSplit s, bool label) const
{
    if (s==DataSplit::TEST)
    {
        if (label)
            return _Ytest;
        else
            return _Xtest;
    }
    else
    {
        if (label)
            return _Ytrain;
        else
            return _Xtrain;
    }
}