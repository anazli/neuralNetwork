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

    train_data = train_data.trans();
    test_data = test_data.trans();

    size_t nx = train_data.rows();

    vector<size_t> layers{nx, 64, 64, 1}; //including the input layer. 
    vector<string> activ{"relu", "relu", "relu"};

    Model regression(layers, activ, "mean_squared_error");
    regression.train(0.1, 32, 500, train_data, train_labels);

    regression.predict(test_data);

    return 0;
}