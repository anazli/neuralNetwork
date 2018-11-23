#include "matrix.h"

double sd{1984.};

std::mt19937 gen(sd);

void seed(const double& s)
{
    sd = s;
}

Matrix<double> real_rand(size_t r, size_t c, double lower, double upper)
{   
    std::uniform_real_distribution<double> rnd(lower, upper);
    Matrix<double> m(r,c);
    for(size_t i = 0 ; i < r ; ++i)
    {
        for(size_t j = 0 ; j < c ; ++j)
        {
            m(i,j) = rnd(gen);
        }
    }

    return m;

}

Matrix<int> int_rand(size_t r, size_t c, size_t lower, size_t upper)
{   
    std::uniform_int_distribution<int> rnd(lower, upper); //including the upper
    Matrix<int> m(r,c);
    for(size_t i = 0 ; i < r ; ++i)
    {
        for(size_t j = 0 ; j < c ; ++j)
        {
            m(i,j) = rnd(gen);
        }
    }

    return m;

}
