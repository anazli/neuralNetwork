#include<iostream>
#include<vector>
#include "matrix.h"
#include "model.h"

using namespace std;


int main()
{
    size_t nx = 16;
    size_t train_size = 16000;
    size_t test_size = 4000;
    size_t Nclasses = 26;

    Matrix<double> train_data(train_size, nx); 
    Matrix<double> train_labels_raw(train_size,1); 
    Matrix<double> test_data(test_size, nx); 
    Matrix<double> test_labels_raw(test_size,1); 

    read_from_file("data/letter_recognition/train_data.dat", train_data);
    read_from_file("data/letter_recognition/train_labels.dat", train_labels_raw);
    read_from_file("data/letter_recognition/test_data.dat", test_data);
    read_from_file("data/letter_recognition/test_labels.dat", test_labels_raw);

    Matrix<double> train_labels(Nclasses, train_size);
    Matrix<double> test_labels(Nclasses, test_size);
    one_hot(train_labels_raw, train_labels);
    one_hot(test_labels_raw, test_labels);

    //training requires data of shape [nx,m]
    train_data = train_data.trans();
    test_data = test_data.trans();

    //Feature normalization
    Matrix<double> mu = mean_value(train_data,0);
    Matrix<double> st = standard_dev(train_data,0);
    mu = extend_rows(mu,nx);
    st = extend_rows(st,nx);

    train_data = (train_data - mu)/st;
    mu = mu.sub_matrix(0,nx,0,test_size);//test set has different m
    st = st.sub_matrix(0,nx,0,test_size);
    test_data = (test_data - mu)/st;

    vector<size_t> layers{nx, 26, 26, Nclasses}; //including the input layer. 
    vector<string> activ{"relu", "relu", "softmax"};

    Model classification(layers, activ, "softmax_loss", "RMSprop");
    classification.train(0.01, 32, 5, train_data, train_labels);

    double train_error = classification.loss;
    cout << "Training error:" << train_error << endl;

    double test_error = classification.evaluate(test_data, test_labels, "accuracy");
    cout << "Test error:" << test_error << endl;

    return 0;
}