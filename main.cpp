#include<iostream>
#include<vector>
#include "matrix.h"
#include "model.h"

using namespace std;


int main()
{
    // Boston housing dataset
    Matrix<double> train_data = read_from_file("data/train_data.dat", 404, 13);
    Matrix<double> train_labels = read_from_file("data/train_labels.dat", 1, 404);
    Matrix<double> test_data = read_from_file("data/test_data.dat", 102, 13);
    Matrix<double> test_labels = read_from_file("data/test_labels.dat", 1, 102);

    train_data = train_data.trans();
    test_data = test_data.trans();

    size_t nx = train_data.rows();

    vector<size_t> layers{nx, 64, 64, 1}; //including the input layer. 
    vector<string> activ{"relu", "relu", "relu"};

    Model regression(layers, activ, "mean_squared_error", "RMSprop");
    regression.train(0.001, 202, 500, train_data, train_labels);
    //performs well on training data when lr=0.001, beta(RMSprop par)=0.95
    //without RMSprop, a lr=0.1 could be used.
    //a good batch size is 202 in order all data to be included.

    double train_error = regression.loss;
    cout << "Training error:" << train_error << endl;

    double test_error = regression.evaluate(test_data, test_labels, "mae");
    cout << "Test error:" << test_error << endl;

    return 0;
}