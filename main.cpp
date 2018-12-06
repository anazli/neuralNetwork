#include<iostream>
#include<vector>
#include "matrix.h"
#include "model.h"

using namespace std;


int main()
{
    size_t nx = 13;
    size_t train_size = 404;
    size_t test_size = 102;
    size_t Nclasses = 1;
    // Boston housing dataset. Data form (m,nx)
    Matrix<double> train_data(train_size, nx);
    read_from_file("data/boston_house/train_data.dat", train_data);
    Matrix<double> train_labels(Nclasses, train_size);
    read_from_file("data/boston_house/train_labels.dat", train_labels);
    Matrix<double> test_data(test_size, nx);
    read_from_file("data/boston_house/test_data.dat", test_data);
    Matrix<double> test_labels(Nclasses, test_size);
    read_from_file("data/boston_house/test_labels.dat", test_labels);

    //data must be of the form (nx,m) for training and evaluation.
    train_data = train_data.trans();
    test_data = test_data.trans();

    vector<size_t> layers{nx, 64, 64, 1}; //including the input layer. 
    vector<string> activ{"relu", "relu", "relu"};

    Model regression(layers, activ, "mean_squared_error", "RMSprop");
    regression.train(0.001, 202, 500, train_data, train_labels);

    double train_error = regression.loss;
    cout << "Training error:" << train_error << endl;

    double test_error = regression.evaluate(test_data, test_labels, "mae");
    cout << "Test error:" << test_error << endl;

    return 0;
}
