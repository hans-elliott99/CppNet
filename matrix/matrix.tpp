
#include "matrix.hpp"
#pragma once

// Default constructor
template <typename T>
Matrix<T>::Matrix(size_t shape_i, size_t shape_j):
        _shape_i(shape_i), 
        _shape_j(shape_j), 
        data(shape_i*shape_j, 0)
{}

/**
 * OPERATORS
*/

template <typename T> T& 
Matrix<T>::operator()(size_t i, size_t j)
{
    ASSERT (i < _shape_i && "`i` index out of range.");
    ASSERT (j < _shape_j && "`j` index out of range.");
    return data[i*_shape_j + j]; 
}

template <typename T> T const& 
Matrix<T>::operator()(size_t i, size_t j) const
{
    ASSERT (i < _shape_i && "`i` index out of range.");
    ASSERT (j < _shape_j && "`j` index out of range.");
    return data[i*_shape_j + j]; 
}


template <typename T> std::vector<T>
Matrix<T>::operator[](size_t i) 
{
    std::vector<T> slice(
        data.begin() + (i*_shape_j),
        data.begin() + (i*_shape_j + _shape_j)
    );
    return slice;
}


template <typename Y> std::ostream& 
operator<<(std::ostream& stream, Matrix<Y>& matrix)
{ 
    stream << '\n';
    for (size_t i {0}; i < matrix._shape_i; i++)
    {
        for (size_t j{0}; j < matrix._shape_j; j++)
            {std::cout << matrix.data[i*matrix._shape_j + j] << " \t";}
        std::cout<<'\n';
    }
    return stream;
}


/**
 * Access subvectors
*/
template <typename T> std::vector<T> 
Matrix<T>::Row(int row, int colBegin, int colEnd) const
{
    if (colEnd < 0) colEnd = _shape_j - 1;
    if (colBegin <= colEnd)
        return _stridedSlice(index(row, colBegin), colEnd-colBegin+1, 1);
    else
        return _stridedSlice(index(row, colBegin), colBegin-colEnd+1, -1);
}


template <typename T> std::vector<T> 
Matrix<T>::Column(int col, int rowBegin, int rowEnd) const
{
    if (rowEnd < 0) rowEnd = _shape_i - 1;
    if (rowBegin <= rowEnd)
        return _stridedSlice(index(rowBegin, col), rowEnd-rowBegin+1, _shape_j);
    else
        return _stridedSlice(index(rowBegin, col), rowBegin-rowEnd+1, -_shape_j);
}


template <typename T> std::vector<T>
Matrix<T>::_stridedSlice(int start, int length, int stride) const
{
    //https://stackoverflow.com/questions/15778377/get-the-first-column-of-a-matrix-represented-by-a-vector-of-vectors
    std::vector<T> out;
    out.reserve(length);
    
    const T* pos = &data[start];
    for (int i {0}; i < length; i++)
    {
        out.push_back(*pos);
        pos += stride; //increase value of pointer to move to next element in vector
    }
    return out;
}

/**
 * std::vector-like methods
*/
template <typename T> size_t 
Matrix<T>::size(size_t axis) const
{
    if (axis == 0)
        return _shape_i;
    else
        return _shape_j;
}


/**
 * Modifiers
*/
template <typename T> Matrix<T>& 
Matrix<T>::add(const Matrix<T>& B)
{
    const size_t Brows = B.size(0);
    const size_t Bcols = B.size(1);
    Matrix<T> _B;

    if (Brows != _shape_i)
    {
        ASSERT (Bcols == _shape_j && (Brows == 1 | Brows == 0) 
                && "Dimensions do not match and could not be broadcasted.");
        _B = B.broadcast(0, _shape_i);
    }
    else if (Bcols != _shape_j)
    {
        ASSERT (Brows == _shape_i && (Bcols == 1 | Bcols == 0) 
                && "Dimensions do not match and could not be broadcasted.");
        _B = B.broadcast(1, _shape_j);
    } 
    else
        { _B = B; }

    // Modify the matrix's data inplace
    std::transform(
        data.begin(), data.end(),
        _B.data.begin(),
        data.begin(),
        std::plus<T>()
    );

    return *this;
}

template <typename T> Matrix<T>& 
Matrix<T>::mul(const Matrix<T>& B)
{
    const size_t Brows = B.size(0);
    const size_t Bcols = B.size(1);
    Matrix<T> _B;

    if (Brows != _shape_i)
    {
        ASSERT (Bcols == _shape_j && (Brows == 1 | Brows == 0) 
                && "Dimensions do not match and could not be broadcasted.");
        _B = B.broadcast(0, _shape_i);
    }
    else if (Bcols != _shape_j)
    {
        ASSERT (Brows == _shape_i && (Bcols == 1 | Bcols == 0) 
                && "Dimensions do not match and could not be broadcasted.");
        _B = B.broadcast(1, _shape_j);
    } 
    else
        { _B = B; }

    // Modify the matrix's data inplace
    std::transform(
        data.begin(), data.end(),
        _B.data.begin(),
        data.begin(),
        std::multiplies<T>()
    );

    return *this;
}


template <typename T> Matrix<T>& 
Matrix<T>::fill(T value)
{
    data = std::vector<T>(_shape_i*_shape_j, value);
    return *this;
}

template <typename T> Matrix<T>& 
Matrix<T>::zero()
{
    this->fill(0);
    return *this;
}

template <typename T> Matrix<T>& 
Matrix<T>::diagonal(T value)
{
    ASSERT (_shape_i == _shape_j); //n x n only
    for (size_t i {0}; i < _shape_i; i++)
        data[i*_shape_j + i] = value;
    
    return *this;
}

template <typename T> Matrix<T>& 
Matrix<T>::randomize(double low, double high)
{
    for (size_t n = 0; n < _shape_i*_shape_j; n++)
    {
        data[n] = matrix::random<T>(low, high);
    }
    
    return *this;
}

template <typename T> Matrix<T>& 
Matrix<T>::transpose()
{
    //https://stackoverflow.com/questions/9227747/in-place-transposition-of-a-matrix
    std::vector<T> result;
    result.reserve(_shape_i*_shape_j);

    #pragma omp parallel for
    for(size_t n {0}; n < _shape_i*_shape_j; n++)
    {
        size_t i = n / _shape_i;
        size_t j = n % _shape_i;
        result.push_back(data[j*_shape_j + i]);
    }

    data = result;
    std::swap(_shape_i, _shape_j);

    return *this;
}


/**
 * APPLY
*/

template <typename T> Matrix<T>& 
Matrix<T>::apply(std::function<T(T)> fun)
{
    std::transform(
        data.begin(), data.end(),
        data.begin(),
        fun
    );

    return *this;
}

template <typename T> Matrix<T>&
Matrix<T>::colApply(T (*fun)(std::vector<T>&))
{
    std::vector<T> result;
    result.reserve(_shape_j);

    std::vector<T> col;
    T value;
    for (size_t j = 0; j < _shape_j; j++)
    {
        col = Column(j); //make fn that returns pointers instead of copying the column's data?
        value = (*fun)(col);   
        result.push_back(value);
    }

    data = result;
    _shape_i = 1;

    return *this;
}

template <typename T> Matrix<T>&
Matrix<T>::rowApply(T (*fun)(std::vector<T>&))
{
    std::vector<T> result;
    result.reserve(_shape_i);

    std::vector<T> row;
    T value;
    for (size_t i = 0; i < _shape_i; i++)
    {
        row = Row(i);
        value = (*fun)(row);
        result.push_back(value);
    }
    data = result;
    _shape_j = 1;

    return *this;
}

/**
 * Broadcasting
*/
template <typename T> Matrix<T> 
Matrix<T>::broadcast(size_t dim, size_t length) const
{
    size_t matsize;     //size of the 1d matrix (ie, length of the row or col vector)
    size_t si, sj;      //the i and j sizes/shapes for the output matrix
    std::vector<T> mat; //the new vector to hold the broadcasted data  

    if (dim == 0) 
    {
        ASSERT ((this->size(0)==1 | this->size(0)==0) 
                    && "The size of the brodacast dimension must be 1 or 0.");
        // For broadcasting along the i dimension, repeat the entire
        // vector length times in the j direction. 
        matsize = this->size(1); //(number of cols in the col vector)
        si = length;
        sj = _shape_j;

        for (size_t i = 0; i < length; i++)
            mat.insert(mat.end(), std::begin(data), std::end(data));
    }
    else
    {
        ASSERT ((this->size(1)==1 | this->size(1)==0) 
            && "The size of the brodacast dimension must be 1 or 0.");

        // For broadcasting along the j dimension, simply repeat the 
        // element in each i position length times in the j direction.
        matsize = this->size(0); //(number of rows in the row vector)
        si = _shape_i;
        sj = length;

        for (size_t i {0}; i < matsize; i++)
            mat.insert(mat.end(), length, data[i]);
    } 

    Matrix<T> Out(si, sj);
    Out.data = mat;
    return Out;
}




/**
 * Utilities
*/
template <typename T> std::vector<size_t>
Matrix<T>::whichTrue(bool (*fun)(T)) const
{
    std::vector<size_t> indices;
    // Where the function evaluates to true, store the index
    for (size_t n = 0; n <_shape_i*_shape_j; n++)
    {
        if ((*fun)(this->data[n]) == true)
        {
            indices.push_back(n);
        }
    }
    return indices;
}


template <typename T> void 
Matrix<T>::print(int nrow, int precis, bool sci)
{
    if (nrow == -1) 
        {nrow = _shape_i; }

    std::cout << std::fixed;
    if (sci)
        std::cout << std::setprecision(precis) << std::scientific;
    else
        std::cout << std::setprecision(precis);

    for (size_t i {0}; i < nrow; i++)
    {
        for (size_t j {0}; j < _shape_j; j++)
            {                 
                std:: cout << data[i*_shape_j + j] << " \t";
            }
        std::cout<<'\n';
    }
}


template <typename T> void 
Matrix<T>::shape()
{
    std::cout << "(" << _shape_i << ", " << _shape_j << ")\n";
}

template <typename T> double
Matrix<T>::mean()
{
    if (data.empty())
        return 0;
    
    double const count = static_cast<double>(data.size());
    return std::reduce(data.begin(), data.end(), 0.0) / count;
}
template <typename T> T
Matrix<T>::sum()
{
    if (data.empty())
        return 0;
    return std::reduce(data.begin(), data.end(), 0.0);
}

template <typename T> double
Matrix<T>::sd()
{
    // Credit : https://stackoverflow.com/questions/7616511/calculate-mean-and-standard-deviation-from-a-vector-of-samples-in-c-using-boos
    double const count = static_cast<double>(data.size());
    double const mu    = static_cast<double>( this->mean() );
    std::vector<double> diff(data.size());

    std::transform(
        data.begin(), data.end(), diff.begin(),
        [mu](double x) { return x - mu; }
    );

    double sq_sum = std::inner_product(diff.begin(), diff.end(), diff.begin(), 0.0);
    
    return std::sqrt(sq_sum / count);
}



/**
 * Namespace functions
*/

template <typename T> Matrix<T> 
matrix::matmul(const Matrix<T>& A, const Matrix<T>& B)
{
    const size_t Arows = A.size(0); const size_t Acols = A.size(1);
    const size_t Brows = B.size(0); const size_t Bcols = B.size(1);
    ASSERT (Acols == Brows);

    Matrix<T> Out(Arows, Bcols);
    // https://stackoverflow.com/questions/16737298/what-is-the-fastest-way-to-transpose-a-matrix-in-c
    Matrix<T> Bt = transpose(B);
    for (size_t i {0}; i < Arows; i++) //rows in a
    {
        for (size_t j {0}; j < Bcols; j++) //cols in b
        {
            T product = 0;
            for (size_t v {0}; v < Acols; v++) //elements in each row
                { product += A.data[i*Acols + v] * Bt.data[j*Brows + v]; }            
               
        Out.data[i*Bcols + j] = product;
        }
    }

    return Out;
}

template <typename T> Matrix<T> 
matrix::matsum(const Matrix<T>& A, const Matrix<T>& B)
{
    ASSERT (A.size(0) == B.size(0)); ASSERT (A.size(1) == B.size(1));

    Matrix<T> Out(A.size(0), A.size(1));
    for (size_t i {0}; i < A.size(0); i++)
    {
        std::transform(
            A.data.begin(), A.data.end(), 
            B.data.begin(),
            Out.data.begin(),
            std::plus<T>()
        );
    }
    return Out;
}

template <typename T> Matrix<T> 
matrix::divide(const Matrix<T>& A, const Matrix<T>& B)
{
    ASSERT (A.size(0) == B.size(0)); ASSERT (A.size(1) == B.size(1));

    Matrix<T> Out(A.size(0), A.size(1));
    for (size_t i {0}; i < A.size(0); i++)
    {
        std::transform(
            A.data.begin(), A.data.end(), 
            B.data.begin(),
            Out.data.begin(),
            std::divides<T>()
        );
    }
    return Out;
}



template <typename T> Matrix<T> 
matrix::transpose(const Matrix<T>& A)
{
    size_t Arows = A.size(0); 
    size_t Acols = A.size(1);
    Matrix<T> Out(Acols, Arows);

    #pragma omp parallel for
    for(size_t n {0}; n < Arows*Acols; n++)
    {
        size_t i = n / Arows;
        size_t j = n % Arows;
        Out.data[n] = A.data[j*Acols + i];
    }
    return Out;
}


template <typename T>
T matrix::vecSum(std::vector<T> &vec)
{
    T sum = 0;
    for (auto& n : vec)
        sum += n;
    return sum;
}

template <typename T>
T matrix::vecMean(std::vector<T> &vec)
{
    T sum = 0;
    for (auto& n : vec)
        sum += n;
    return (sum / static_cast<T>(vec.size()));
}

template <typename T>
T matrix::mean(Matrix<T> &A)
{
    T sum = 0;
    for (auto& n : A.data)
        sum += n;
    return (sum / static_cast<T>(A.nelements()));
}


template <typename T>
T matrix::random(double low, double high)
{
    return low + static_cast<T>(rand()) / ( static_cast<T>(RAND_MAX / (high - low)) );
}