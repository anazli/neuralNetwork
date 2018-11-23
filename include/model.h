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

    Model(const std::vector<size_t>& l,
          const std::vector<std::string>& act,
          std::string name)
    {
        //set_parameters(layers, m);
        layers = l;
        activations = act;
        loss_function = name;
        loss = 0.;
    }

    void set_parameters(size_t m);

    void forward_prop(const Matrix<double> data);

    void backward_prop(const Matrix<double>& data,
                       const Matrix<double>& labels);

    void gradient_descent(const double& learning_rate); 

    void train(const double& learning_rate,
               size_t batch_size, size_t epochs,
               const Matrix<double>& train_data,
               const Matrix<double>& train_labels);

    void predict(const Matrix<double>& test_data);

    std::vector< Matrix<double> > weights;
    std::vector< Matrix<double> > biases;
    std::vector< Matrix<double> > zetas;
    std::vector< Matrix<double> > alphas;
    std::vector< Matrix<double> > dw;
    std::vector< Matrix<double> > db;
    std::vector<size_t> layers;
    std::vector<std::string> activations;
    std::string loss_function;
    double loss;

};

double sigmoid(double);
double sigmoid_deriv(double);
double relu(double);
double relu_deriv(double);
double cross_entropy(const Matrix<double>& y, const Matrix<double>& a);
Matrix<double> cross_entropy_deriv(const Matrix<double>& y, const Matrix<double>& a);
double mean_squared_error(const Matrix<double>& y, const Matrix<double>& a);
Matrix<double> mse_deriv(const Matrix<double>& y, const Matrix<double>& a);
Matrix<double> read_from_file(const std::string& file, size_t r, size_t c);

#endif