#ifndef MODEL_H
#define MODEL_H

#include<algorithm>
#include<iostream>
#include<fstream>
#include<string>
#include<cmath>
#include<vector>
#include "matrix.h"


class Model {
    
public:

    Model(const std::vector<size_t>&,
          const std::vector<std::string>&,
          const std::string&,
          const std::string&);

    void set_parameters(size_t);
    void forward_prop(const Matrix<double>&);
    void backward_prop(const Matrix<double>&, const Matrix<double>&);
    void gradient_descent(const double&); 
    void gradient_descent_with_RMSprop(const double&); 
    void train(const double&, size_t, size_t,
               const Matrix<double>&, const Matrix<double>&);
    void predict(const Matrix<double>&);

    std::vector< Matrix<double> > weights;
    std::vector< Matrix<double> > biases;
    std::vector< Matrix<double> > zetas;
    std::vector< Matrix<double> > alphas;
    std::vector< Matrix<double> > dw;
    std::vector< Matrix<double> > db;
    std::vector< Matrix<double> > Sdw;
    std::vector< Matrix<double> > Sdb;
    std::vector<size_t> layers;
    std::vector<std::string> activations;
    std::string loss_function;
    std::string optimizer;
    double loss;
};

double sigmoid(double);
double sigmoid_deriv(double);
double relu(double);
double relu_deriv(double);
double cross_entropy(const Matrix<double>&, const Matrix<double>&);
Matrix<double> cross_entropy_deriv(const Matrix<double>&, const Matrix<double>&);
double mean_squared_error(const Matrix<double>&, const Matrix<double>&);
Matrix<double> mse_deriv(const Matrix<double>&, const Matrix<double>&);
double glorot_uniform(size_t, size_t);
Matrix<double> read_from_file(const std::string&, size_t, size_t); 
Matrix<double> normalize_data(const Matrix<double>&, size_t);

#endif