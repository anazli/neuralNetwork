#ifndef MATRIX_H
#define MATRIX_H

#include<iostream>
#include<fstream>
#include<string>
#include<random>
#include<vector>

template <class T>
class Matrix {

public:
    
    Matrix(size_t r = 0, size_t c = 0, T elem = 0);

    size_t rows()const{return m_matrix.size();}
    size_t cols()const{return !(m_matrix.empty())? m_matrix[0].size(): m_matrix.size();}

    T& operator()(size_t i, size_t j){return m_matrix[i][j];}
    const T& operator()(size_t i, size_t j)const{return m_matrix[i][j];}
    Matrix<T> operator-()const;

    void shape()const{std::cout << "(" << rows() << "," << cols() << ")\n";}
    void print()const;
    void print_to_file(std::string file)const;
    Matrix<T> sub_matrix(size_t rs, size_t re, size_t cs, size_t ce)const;
    Matrix<T> trans()const;

private:
    std::vector< std::vector<T> > m_matrix;

};

void seed(const double&);
Matrix<double> real_rand(size_t r, size_t c, double lower = 0., double upper = 1.);
Matrix<int> int_rand(size_t r, size_t c, size_t lower, size_t upper);

template <typename T>
Matrix<T> apply_function(const Matrix<T>& m, T (*f)(T))
{
    size_t r = m.rows();
    size_t c = m.cols();
    Matrix<T> ret(r,c);
    for(size_t i = 0 ; i < r ; ++i)
    {
        for(size_t j = 0 ; j < c ; ++j)
        {
            ret(i,j) = f( m(i,j) );
        }
    } 

    return ret;
}

template <typename T>
Matrix<T> sum(const Matrix<T>& m, int axis = 0)
{
    size_t r = m.rows();
    size_t c = m.cols();
    if(axis == 0) //sum of all rows
    {
        Matrix<T> ret(1,c);
        for(size_t i = 0 ; i < c ; ++i)
        {
            for(size_t j = 0 ; j < r ; ++j)
            {
                ret(0,i) += m(j,i); //sum of every row in a given column
            }
        }

        return ret;
    }
    if(axis == 1) //sum of all cols
    {
        Matrix<T> ret(r,1);
        for(size_t i = 0 ; i < r ; ++i)
        {
            for(size_t j = 0 ; j < c ; ++j)
            {
                ret(i,0) += m(i,j); //sum of every col in a given row
            }
        }

        return ret;
    }

    std::cout << "Please specify the axis! 0 = sum of rows, 1 = sum of columns.\n";
    throw "Sumation error!\n";
}

// Constructor with default arguments. We can create a (i,j) matrix or
// an empty matrix. There is no push_back operation so the only reason
// the later is created is when we want to copy an existing matrix (i,j)
// to it or when it's asigned the result of an operation (+,-,*) between 2 matrices.
template <typename T>
Matrix<T>::Matrix(size_t r, size_t c, T elem)
{

    int ir = static_cast<int>(r); //because they are size_t, a negative int
    int ic = static_cast<int>(c); // input will change to unsigned long which 
    if(ir < 0 || ic < 0)          // gives a big positive number.
    {
        std::cout << "Invalid size:" << "(" << ir << "," << ic << ") for a Matrix!\n";
        throw "Dimension error!";
    }

    std::vector< std::vector<T> > temp(r, std::vector<T>(c, elem));// instead of a 2 for loops
    m_matrix = temp;                                            // create a copy of a 2D vector

}

// Unary operator (-).
template <typename T>
Matrix<T> Matrix<T>::operator-()const
{
    size_t r = this->rows();
    size_t c = this->cols();
    Matrix<T> ret(r,c);
    for(size_t i = 0 ; i < r ; ++i)
    {
        for(size_t j = 0 ; j < c ; ++j)
        {
            ret(i,j) = -this->operator()(i,j);
            
        }
    }

    return ret;
}

template <typename T>
void Matrix<T>::print()const
 {
    size_t r = rows();
    size_t c = cols();

    for(size_t i = 0 ; i < r ; ++i)
    {
        for(size_t j = 0 ; j < c ; ++j)
        {
            std::cout << m_matrix[i][j] << " "; 
        }
        std::cout << std::endl;
    }
    std::cout << "---\n";
}


template <typename T>
void Matrix<T>::print_to_file(std::string file)const
 {
    size_t r = rows();
    size_t c = cols();
    std::ofstream out;
    out.open(file);

    for(size_t i = 0 ; i < r ; ++i)
    {
        for(size_t j = 0 ; j < c ; ++j)
        {
            out << m_matrix[i][j] << " "; 
        }
        out << std::endl;
    }
    out.close();
}

template <typename T>
Matrix<T> Matrix<T>::sub_matrix(size_t r1, size_t r2, size_t c1, size_t c2)const
{
    if(static_cast<int>(r1) > static_cast<int>(r2) ||
       static_cast<int>(c1) > static_cast<int>(c2))
    {
        std::cout << "Invalid intervals for a sub-matrix!\n";
        throw "Interval error!\n";
    }
    //When one argument is negative, we want an Index out of bounds exception.
    //that's why I use static_cast<int>, otherwise the first exception is thrown.
    if( (static_cast<int>(r1) < 0 || static_cast<int>(r2) > this->rows()) ||
        (static_cast<int>(c1) < 0 || static_cast<int>(c2) > this->cols()) )
    {
        std::cout << "Index out of  bounds!\n";
        throw "Out of bounds error!\n";
    }

    size_t r = r2 - r1;
    size_t c = c2 - c1;
    size_t sub_i = 0; //the indexes of the sub_matrix must start from 0
    size_t sub_j = 0;
    if(r == 0 && c != 0) //not just r == 0, because this will run even when c == 0 is also true.
    {
        Matrix<T> m(1,c);
        //when we want take a matrix's row (r==0), then
        //the sub-matrix is a (1,c) column vector.
        sub_j = 0;
        for(size_t i = c1 ; i < c2 ; ++i)
        {
            m(r,sub_j++) = this->operator()(r1,i);
        }
        return m;
    }
    else if(c == 0 && r != 0) //same as the first condition.
    {
        Matrix<T> m(r,1);
        sub_i = 0;
        for(size_t i = r1 ; i < r2 ; ++i)
        {
            m(sub_i++,c) = this->operator()(i,c1);
        }
        return m;
    }
    else if(r == 0 && c == 0)//this condition is because when r1==r2, c1==c2
    {                        //the 2 for loops in the following contition
        Matrix<T> m(1,1);    //will not run and the shape of the sub-m will 
        m(0,0) = this->operator()(r1,c1);//be (0,0) (empty matrix) and not (1,1)->scalar value.
        return m;
    }
    else 
    {
        Matrix<T> m(r,c);
        sub_i = 0;
        for(size_t i = r1 ; i < r2 ; ++i)
        {
            sub_j = 0;
            for(size_t j = c1 ; j < c2 ; ++j)
            {
                m(sub_i,sub_j) = this->operator()(i,j);
                sub_j++;
            }
            sub_i++;
        }
        return m;
    }

}

template <typename T>
Matrix<T> Matrix<T>::trans()const
{
    size_t r = rows();
    size_t c = cols();
    Matrix<T> m(c,r);
    for(size_t i = 0 ; i < r ; ++i)
    {
        for(size_t j = 0 ; j < c ; ++j)
        {
            m(j,i) = this->operator()(i,j);
        }
    }

    return m;
}


// Matrix multiplication.
template <typename T>
Matrix<T> multiply(const Matrix<T>& m1, const Matrix<T>& m2)
{
    size_t rows1 = m1.rows();
    size_t cols1 = m1.cols();

    size_t rows2 = m2.rows();
    size_t cols2 = m2.cols();

    if(cols1 != rows2)
    {
        std::cout << "Cannot multiply a (" << rows1 << "," << cols1 << 
                ") matrix with a (" << rows2 << "," << cols2 << ") matrix\n";
        throw "Multiplication error!";
    }

    Matrix<T> m(rows1, cols2);
    
    for(size_t i = 0 ; i < rows1 ; ++i)
    {
        for(size_t j = 0 ; j < cols2 ; ++j)
        {
            T sum{0};
            for(size_t k = 0 ; k < rows2 ; ++k)
            {
                sum += m1(i,k) * m2(k,j); 
            }
            m(i,j) = sum;
        }
    }

    return m;
}

template <typename T>
Matrix<T> extend_rows(const Matrix<T>& m, size_t r)
{
    size_t c = m.cols();
    Matrix<T> ret(r,c);
    for(size_t i = 0 ; i < r ; ++i)
    {
        for(size_t j = 0 ; j < c ; ++j)
        {
            ret(i,j) = m(0,j);
        }
    }
    return ret;
}

template <typename T>
Matrix<T> extend_cols(const Matrix<T>& m, size_t c)
{
    size_t r = m.rows();
    Matrix<T> ret(r,c);
    for(size_t i = 0 ; i < r ; ++i)
    {
        for(size_t j = 0 ; j < c ; ++j)
        {
            ret(i,j) = m(i,0);
        }
    }
    return ret;
}
//---------Comparison operators------------------------
// Two types (T, S), they don't have to be of the same type
// because we just compare their dimensions.
template <typename T, typename S>
bool operator==(const Matrix<T>& m1, const Matrix<S>& m2)
{
    return (m1.rows() == m2.rows() && m1.cols() == m2.cols());
}

template <typename T, typename S>
bool operator!=(const Matrix<T>& m1, const Matrix<S>& m2)
{
    return !(m1 == m2);
}

//---------Arithmetic operators------------------------

//Addition of two Matrices.
template <typename T>
Matrix<T> operator+(const Matrix<T>& m1, const Matrix<T>& m2)
{
    if(m1 != m2)
    {
        std::cout << "Cannot add two Matrices with different shapes!\n";
        throw "Addition error.\n";
    }

    size_t r = m1.rows();
    size_t c = m1.cols();
    Matrix<T> m(r, c);
    for(size_t i = 0 ; i < r ; ++i)
    {
        for(size_t j = 0 ; j < c ; ++j)
        {
            m(i,j) = m1(i,j) + m2(i,j);
        }
    }

    return m;
}

//Addition of a Matrix and number.
template <typename T>
Matrix<T> operator+(const Matrix<T>& m, const T& num)
{
    size_t r = m.rows();
    size_t c = m.cols();
    Matrix<T> ret(r,c);
    for(size_t i = 0 ; i < r ; ++i)
    {
        for(size_t j = 0 ; j < c ; ++j)
        {
            ret(i,j) = m(i,j) + num;
        }
    }

    return ret;
}

//Addition of a number and Matrix using the operator above.
template <typename T>
Matrix<T> operator+(const T& num, const Matrix<T>& m)
{
    return m + num;
}

// Subtractions of two Matrices.
template <typename T>
Matrix<T> operator-(const Matrix<T>& m1, const Matrix<T>& m2)
{
    if(m1 != m2)
    {
        std::cout << "Cannot subtract two Matrices with different shapes!\n";
        throw "Subtraction error.\n";
    }
    return m1 + (-m2);
}

template <typename T>
Matrix<T> operator-(const Matrix<T>& m, const T& num)
{
    return m + (-num);
}

template <typename T>
Matrix<T> operator-(const T& num, const Matrix<T>& m)
{
    return num + (-m);
}

// Element-wise multiplication (Matrix, Matrix), (Matrix, scalar)
// Using the overloaded * operator.
template <typename T>
Matrix<T> operator*(const Matrix<T>& m1, const Matrix<T>& m2)
{
    Matrix<T> temp(1,1); 
    if(m1 == temp) //if one matrix is (1,1)=scalar do matrix-scalar element-wise multiplication.
    {
        return m1(0,0) * m2;
    }
    else if(m2 == temp) // the same.
    {
        return m1 * m2(0,0);
    }
    else if(m1 == temp && m2 == temp) //if they are both (1,1) return a (1,1) matrix.
    {
        return m1(0,0) * m2(0,0);
    }

    if(m1 != m2) //if they are other than (1,1) and their dims don't match, throw.
    {
        std::cout << "Matrix dimensions don't match. Cannot do element-wise multiplication!\n";
        throw "Element-wise multiplication error.\n";
    }
    size_t r = m1.rows();
    size_t c = m1.cols();
    Matrix<T> m(r,c);
    for(size_t i = 0 ; i < r ; ++i)
    {
        for(size_t j = 0 ; j < c ; ++j)
        {
            m(i,j) = m1(i,j) * m2(i,j);
        }
    }
    
    return m;
}

template <typename T>
Matrix<T> operator*(const Matrix<T>& m, const T& num)
{
    size_t r = m.rows();
    size_t c = m.cols();
    Matrix<T> ret(r,c);
    for(size_t i = 0 ; i < r ; ++i)
    {
        for(size_t j = 0 ; j < c ; ++j)
        {
            ret(i,j) = m(i,j) * num;
        }
    }

    return ret;
}

template <typename T>
Matrix<T> operator*(const T& num, const Matrix<T>& m)
{
    return m * num;
}




// Element-wise division (Matrix, Matrix), (Matrix, scalar)
// Using the overloaded / operator.
template <typename T>
Matrix<T> operator/(const Matrix<T>& m1, const Matrix<T>& m2)
{
    Matrix<T> temp(1,1); 
    if(m1 == temp) //if one matrix is (1,1)=scalar do matrix-scalar element-wise multiplication.
    {
        std::cout << "Cannot divide a scalar by a matrix!\n";
        throw "Element-wise division error.\n";
    }
    else if(m2 == temp) // the same.
    {
        return m1 / m2(0,0);
    }
    else if(m1 == temp && m2 == temp) //if they are both (1,1) return a (1,1) matrix.
    {
        return m1(0,0) / m2(0,0);
    }

    if(m1 != m2) //if they are other than (1,1) and their dims don't match, throw.
    {
        std::cout << "Matrix dimensions don't match. Cannot do element-wise division!\n";
        throw "Element-wise division error.\n";
    }
    size_t r = m1.rows();
    size_t c = m1.cols();
    Matrix<T> m(r,c);
    for(size_t i = 0 ; i < r ; ++i)
    {
        for(size_t j = 0 ; j < c ; ++j)
        {
            m(i,j) = m1(i,j) / m2(i,j);
        }
    }
    
    return m;
}


template <typename T>
Matrix<T> operator/(const Matrix<T>& m, const T& num)
{
    size_t r = m.rows();
    size_t c = m.cols();
    Matrix<T> ret(r,c);
    for(size_t i = 0 ; i < r ; ++i)
    {
        for(size_t j = 0 ; j < c ; ++j)
        {
            ret(i,j) = m(i,j) / num;
        }
    }

    return ret;
}


#endif
