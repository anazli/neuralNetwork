#ifndef MATRIX_H
#define MATRIX_H

#include<iostream>
#include<fstream>
#include<string>
#include<random>
#include<vector>


/********************************************************************
 * 
 * DECLARATIONS  
 * 
 ********************************************************************/



template <class T>
class Matrix {

public:
    
    Matrix(size_t r = 0, size_t c = 0, T elem = 0);

    size_t rows()const{return m_matrix.size();}/*!< \brief Magnetic field's vector coords.*/
    size_t cols()const{return !(m_matrix.empty()) ?
                              m_matrix[0].size() : m_matrix.size();}
                              //if rows=0, cols=0

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



/********************************************************************
 * 
 * GENERAL NON-MEMBER FUNCTIONS  
 * 
 ********************************************************************/



// Defined in matrix.cpp
void seed(const double&);
Matrix<double> real_rand(size_t, size_t, double lower = 0., double upper = 1.);
Matrix<int> int_rand(size_t, size_t, size_t, size_t);



/*! \brief Applies an operation on every element as defined by *f.  
 *
 *  @param Input matrix which the operation is applied on. 
 *  @param Function of type T that takes a T arg and defines the operation.  
 *  @return A new matrix of the same type as the input. 
 */

template <typename T>
Matrix<T> apply_function(const Matrix<T>& m, T (*f)(T))
{

    if(f == nullptr)
    {
        std::cout << "Enter a valid function![apply_function]\n";
        throw "apply_function error.\n";
    }

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
Matrix<T> apply_function(const Matrix<T>& m,
                         Matrix<T> (*f)(const Matrix<T>&))
{

    if(f == nullptr)
    {
        std::cout << "Enter a valid function![apply_function]\n";
        throw "apply_function error.\n";
    }

    Matrix<double> ret = f(m);

    return ret;
}



template <typename T>
Matrix<T> apply_function(const Matrix<T>& m1, const Matrix<T>& m2,
                         Matrix<T> (*f)(const Matrix<T>&, const Matrix<T>&))
{

    if(f == nullptr)
    {
        std::cout << "Enter a valid function![apply_function]\n";
        throw "apply_function error.\n";
    }

    Matrix<double> ret = f(m1,m2);

    return ret;
}



/*! \brief Returns the sum of the rows or columns of a matrix.  
 *
 *  @param Input matrix which the operation is applied on. 
 *  @param size_t arg. If 0 the sum of rows is computed, if 1 the sum of
 *         columns. Default value is 0.  
 *  @return A new matrix of the same type as the input containing the sums.
 *          If axis=0 it's a (1,c) row vector,
 *          if axis=1 it's (r,1) column vector. 
 */

template <typename T>
Matrix<T> sum(const Matrix<T>& m, size_t axis = 0)
{
    if(axis < 0 || axis > 1)
    {
        std::cout << "Wrong axis, enter (0 or 1)![sum]\n";
        throw "Sum error.\n";
    }
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
}



/*! \brief Computes the mean value of the rows or columns of a matrix.  
 *
 *  @param Input matrix which the operation is applied on. 
 *  @param size_t arg. If 0 the mean of rows is computed, if 1 the mean of
 *  columns. Default value is 0.  
 *  @return A new matrix of the same type as the input containing the mean
 *          values. If axis=0 it's a (1,c) row vector, 
 *          if axis=1 it's (r,1) column vector. 
 */

template <typename T>
Matrix<T> mean_value(const Matrix<T>& m, size_t axis = 0)
{
    if(axis < 0 || axis > 1)
    {
        std::cout << "Wrong axis, enter (0 or 1)![mean_value]\n";
        throw "mean_value error.\n";
    }

    size_t r = m.rows();
    size_t c = m.cols();

    if(axis == 0)
    {
        Matrix<T> ret(1,c);
        ret = sum(m,0); 
        ret = ret/(double)r;
        return ret;
    }
    else if(axis == 1)
    {
        Matrix<T> ret(r,1);
        ret = sum(m,1);
        ret = ret/(double)c;
        return ret;
    }
}



/*! \brief Computes the standard deviation of the rows or columns of a matrix.  
 *
 *  @param Input matrix which the operation is applied on. 
 *  @param size_t arg. If 0 the mean of rows is computed, if 1 the mean of
 *  columns. Default value is 0.  
 *  @return A new matrix of the same type as the input containing the std
 *          values. If axis=0 it's a (1,c) row vector, 
 *          if axis=1 it's (r,1) column vector. 
 */

template <typename T>
Matrix<T> standard_dev(const Matrix<T>& m, size_t axis = 0)
{
    if(axis < 0 || axis > 1)
    {
        std::cout << "Wrong axis, enter (0 or 1)![standard_dev]\n";
        throw "standard_dev error.\n";
    }

    size_t r = m.rows();
    size_t c = m.cols();

    if(axis == 0)
    {
        Matrix<T> ret(1,c);
        Matrix<T> mu = mean_value(m,0);
        mu = extend_rows(mu,r);
        ret = (1./(r-1)) * (m - mu) * (m - mu);
        ret = apply_function(ret, sqrt);
        return ret;
    }
    if(axis == 1)
    {
        Matrix<T> ret(r,1);
        Matrix<T> mu = mean_value(m,1);
        mu = extend_cols(mu,c);
        ret = (1./(c-1)) * (m - mu) * (m - mu);
        ret = apply_function(ret, sqrt);
        return ret;
    }
}



/*! \brief Returns the index of the maximum element of a row or column vector.  
 *
 *  @param Input matrix(row or column vector) which the operation is applied on. 
 *  @return A size_t value which is the index of the maximum element. 
 */
template <typename T>
size_t arg_max(const Matrix<T>& m)
{
    size_t r = m.rows();
    size_t c = m.cols();
    size_t max_id{0};
    T max{0};

    if(r == 1 && c != 1)
    {
       for(size_t i = 0 ; i < c ; ++i)
       {
           if(max < m(0,i))
           {
               max = m(0,i);
               max_id = i; 
           }
       } 
    }
    else if(c == 1 && r != 1)
    {
       for(size_t i = 0 ; i < r ; ++i)
       {
           if(max < m(i,0))
           {
               max = m(i,0);
               max_id = i; 
           }
       } 
    }
    return max_id;
}



// Reads the input data and returns an array (r,c). The rows and cols
// must be known in advance. Training requires the data to be of the form
// (nx,m)
template <typename T>
void read_from_file(std::string file, Matrix<T>& m) 
{   
    size_t r = m.rows();
    size_t c = m.cols();
    std::ifstream in;
    in.open(file);
    for(size_t i = 0 ; i < r ; ++i)
    {
        for(size_t j = 0 ; j < c ; ++j)
        {
            T x;
            in >> x;
            m(i,j) = x;
        }
    }
    in.close();
}



template <typename T>
void one_hot(const Matrix<T>& v, Matrix<T>& mat)
{
    size_t r = mat.rows();
    size_t c = mat.cols();//assumes v is column vector. so c = 1
    for(size_t i = 0 ; i < c ; ++i) //for every example
    {
        for(size_t j = 0 ; j < r ; ++j) //assign 1 to the v(i,0)th row element 
        {                               //and zero elsewhere
            if(v(i,0) == j)
                mat(j,i) = 1.;
            else
                mat(j,i) = 0.;
        }
    }
}



/********************************************************************
 * 
 * MEMBER FUNCTIONS  
 * 
 ********************************************************************/



/*! \brief Default Constructor.  
 *
 *  @param size_t r. The number of rows of the matrix (default). 
 *  @param size_t r. The number of cols of the matrix (default). 
 *  @param T elem. A value of type T which is assigned to all elements (default). 
 */

template <typename T>
Matrix<T>::Matrix(size_t r, size_t c, T elem)
{

    int ir = static_cast<int>(r); //because they are size_t, a negative int
    int ic = static_cast<int>(c); // input will change to unsigned long which 
    if(ir < 0 || ic < 0)          // gives a big positive number.
    {
        std::cout << "Invalid size:" << "(" << ir << "," << ic << ")"
                                      " for a Matrix![Constructor]\n";
        throw "Constructor error!";
    }

    std::vector< std::vector<T> > temp(r, std::vector<T>(c, elem));
    m_matrix = temp; //instead of 2 for loops, create a copy of a 2D vector.

}



/*! \brief Unary operator (-). Returns the opposite matrix. */

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



/*! \brief Prints the elements of the matrix to standard output.*/

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



/*! \brief Prints the elements of the matrix to a file.  
 *
 *  @param string file. The name of the file. 
 */

template <typename T>
void Matrix<T>::print_to_file(std::string file)const
 {
    if(file == " ")
    {
        std::cout << "Enter a valid file name![print_to_file]\n";
        throw "print_to_file error.\n";
    }
     
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



/*! \brief Returns a sub-matrix.  
 *
 *  @param size_t r1. First row of the sub-matrix is the r1th row of the present.  
 *  @param size_t r2. Last row of the sub-matrix is the (r2-1)th row of the present. 
 *  @param size_t c1. First col of the sub-matrix is the c1th col of the present.  
 *  @param size_t c2. Last col of the sub-matrix is the (c2-1)th row of the present. 
 *  @return A new matrix of the same type as the present and of shape 
 *          [r2-r1,c2-c1]. 
 */

template <typename T>
Matrix<T> Matrix<T>::sub_matrix(size_t r1, size_t r2, size_t c1, size_t c2)const
{
    if(static_cast<int>(r1) > static_cast<int>(r2) ||
       static_cast<int>(c1) > static_cast<int>(c2))
    {
        std::cout << "Invalid interval![sub_matrix]\n";
        throw "Interval error!\n";
    }
    //When one argument is negative, we want an Index out of bounds exception.
    //that's why I use static_cast<int>, otherwise the first exception is thrown.
    if( (static_cast<int>(r1) < 0 || static_cast<int>(r2) > this->rows()) ||
        (static_cast<int>(c1) < 0 || static_cast<int>(c2) > this->cols()) )
    {
        std::cout << "Index out of  bounds![sub_matrix]\n";
        throw "Out of bounds error!\n";
    }

    size_t r = r2 - r1;
    size_t c = c2 - c1;
    size_t sub_i = 0; //the indexes of the sub_matrix must start from 0
    size_t sub_j = 0;
    if(r == 0 && c != 0) //not just r == 0, because this will run even when
    {                             //c == 0 is also true.
        Matrix<T> m(1,c);
        //when we want to take a matrix's row (r1==r2 hence r==0), then
        //the sub-matrix is a (1,c) row vector.
        sub_j = 0;
        for(size_t i = c1 ; i < c2 ; ++i)
        {
            m(r,sub_j++) = this->operator()(r1,i);
        }
        return m;
    }
    else if(c == 0 && r != 0) //same as the first condition for a column vector.
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
    {                        //the 2 for loops in the next contition
        Matrix<T> m(1,1);    //will not run and the shape of the sub-m will 
        m(0,0) = this->operator()(r1,c1);//be (0,0) (empty matrix) and not (1,1)
        return m;                                              //->scalar value.
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



/*! \brief Returns the transpose matrix of the present matrix.  
 *
 *  @return A new matrix of the same type as the present and of shape [c,r]. 
 */
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



/*! \brief Returns the product of two matrices.  
 *
 *  @param Matrix<T> m1. The first matrix.  
 *  @param Matrix<T> m2. The second matrix.  
 *  @return A new matrix of the same type as the two inputs and of shape [r1,c2]. 
 */

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
                ") matrix with a (" << rows2 << "," << cols2 << ") matrix."
                                                            " [multiply]\n";
        throw "multiply error!";
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



/*! \brief Returns a matrix which has r rows with the same elements as the
 *    input vector at every column. Assuming every column has the same elements.    
 *
 *  @param Matrix<T> m. The input matrix.  
 *  @param size_t r. The number of rows of the new matrix.  
 *  @return A new matrix of the same type as the input and of shape [r,c]. 
 */

template <typename T>
Matrix<T> extend_rows(const Matrix<T>& m, size_t r)
{
    if(static_cast<int>(r) < 1)
    {
        std::cout << "Enter a valid number of rows![extend_rows]\n";
        throw "extend_rows error!\n";
    }
    //actually, it's possible to get a smaller matrix with r < m.rows()
    size_t c = m.cols();
    Matrix<T> ret(r,c);
    for(size_t i = 0 ; i < r ; ++i)
    {
        for(size_t j = 0 ; j < c ; ++j)
        {
            ret(i,j) = m(0,j); //assuming every row of m has the same elements
        }             //as those of the first one. just take those of the first.
    }
    return ret;
}



/*! \brief Returns a matrix which has c cols with the same elements as the
 *         input vector at every row. Assuming every row has the same elements.   
 *
 *  @param Matrix<T> m. The input matrix.  
 *  @param size_t r. The number of cols of the new matrix.  
 *  @return A new matrix of the same type as the input and of shape [r,c]. 
 */

template <typename T>
Matrix<T> extend_cols(const Matrix<T>& m, size_t c)
{
    if(static_cast<int>(c) < 1)
    {
        std::cout << "Enter a valid number of columns![extend_cols]\n";
        throw "extend_rows error!\n";
    }
    //actually, it's possible to get a smaller matrix with c < m.cols()
    size_t r = m.rows();
    Matrix<T> ret(r,c);
    for(size_t i = 0 ; i < r ; ++i)
    {
        for(size_t j = 0 ; j < c ; ++j)
        {
            ret(i,j) = m(i,0); //assuming every col of m has the same elements
        }             //as those of the first one. just take those of the first.
    }
    return ret;
}



/********************************************************************
 * 
 * BINARY OPERATORS  
 * 
 ********************************************************************/



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



/*! \brief Element-wise addition of two Matrices.*/

template <typename T>
Matrix<T> operator+(const Matrix<T>& m1, const Matrix<T>& m2)
{
    if(m1 != m2)
    {
        std::cout << "Cannot add two Matrices with different shapes![+]\n";
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



/*! \brief Element-wise addition of a Matrix and a scalar.*/

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



/*! \brief Element-wise addition of a scalar and a Matrix.*/

template <typename T>
Matrix<T> operator+(const T& num, const Matrix<T>& m)
{
    return m + num;
}



/*! \brief Element-wise subtraction of two Matrices.*/

template <typename T>
Matrix<T> operator-(const Matrix<T>& m1, const Matrix<T>& m2)
{
    if(m1 != m2)
    {
        std::cout << "Cannot subtract two Matrices with different shapes![-]\n";
        throw "Subtraction error.\n";
    }
    return m1 + (-m2);
}



/*! \brief Element-wise subtraction of a Matrix and a scalar.*/

template <typename T>
Matrix<T> operator-(const Matrix<T>& m, const T& num)
{
    return m + (-num);
}



/*! \brief Element-wise subtraction of a scalar and a Matrix.*/

template <typename T>
Matrix<T> operator-(const T& num, const Matrix<T>& m)
{
    return num + (-m);
}



/*! \brief Element-wise multiplication of two Matrices.*/

template <typename T>
Matrix<T> operator*(const Matrix<T>& m1, const Matrix<T>& m2)
{
    Matrix<T> temp(1,1); 
    if(m1 == temp && m2 != temp) //if one matrix is (1,1)=scalar do matrix-scalar
    {              //element-wise multiplication.
        return m1(0,0) * m2;
    }
    else if(m2 == temp && m1 != temp) // the same.
    {
        return m1 * m2(0,0);
    }
    else if(m1 == temp && m2 == temp) //if they are both (1,1) return a (1,1)
    {                                                               // matrix.
        return m1(0,0) * m2(0,0);
    }

    if(m1 != m2) //if they are other than (1,1) and their dims don't match, throw.
    {
        std::cout << "Matrix dimensions don't match. Cannot do element-wise"
                                                        " multiplication!\n";
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



/*! \brief Element-wise multiplication of a Matrix and a scalar.*/

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



/*! \brief Element-wise multiplication of a scalar and a Matrix.*/

template <typename T>
Matrix<T> operator*(const T& num, const Matrix<T>& m)
{
    return m * num;
}



/*! \brief Element-wise division of two Matrices.*/

template <typename T>
Matrix<T> operator/(const Matrix<T>& m1, const Matrix<T>& m2)
{
    Matrix<T> temp(1,1); 
    if(m1 == temp && m2 != temp) //if the first matrix is [1,1]=scalar, throw. 
    {
        std::cout << "Cannot divide a scalar by a matrix!\n";
        throw "Element-wise division error.\n";
    }
    else if(m1 != temp && m2 == temp) // if only the second is [1,1]=scalar, 
    {                      //do element-wise division between matrix and scalar. 
        return m1 / m2(0,0);
    }
    else if(m1 == temp && m2 == temp) //if they are both [1,1] return a [1,1]
    {                                                            // matrix.
        return m1(0,0) / m2(0,0);
    }

    if(m1 != temp && m2 != temp && m1 != m2) //if their dims don't match, throw.
    {
        std::cout << "Matrix dimensions don't match. Cannot do element-wise"
                                                             " division!\n";
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



/*! \brief Element-wise division of a Matrix and a scalar.*/

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
