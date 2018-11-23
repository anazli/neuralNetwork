#include "model.h"

using std::vector;
using std::max;
using std::ifstream;
using std::ofstream;
using std::endl;


double sigmoid(double x)
{
    return 1./(1. + exp(-x));
}

double sigmoid_deriv(double x)
{
    return sigmoid(x) * (1. - sigmoid(x)); 
}

double relu(double x)
{
    return max(0.,x);
}

double relu_deriv(double x)
{
    if(x < 0)
        return 0.;
    else if(x > 0)
        return 1.;

    return NAN;
}

double cross_entropy(const Matrix<double>& y, const Matrix<double>& a)
{
   Matrix<double> l = -(multiply(y, apply_function(a,log).trans())
                         + multiply((1. - y), apply_function(1. - a, log).trans()));

    double ret = l(0,0);
    return ret;
}

Matrix<double> cross_entropy_deriv(const Matrix<double>& y, const Matrix<double>& a)
{
    Matrix<double> ret;
    ret = -y/a + (1.-y)/(1.-a);
    return ret;
}

double mean_squared_error(const Matrix<double>& y, const Matrix<double>& a)
{
    Matrix<double> s = y - a;
    s = s * s;
    Matrix<double> ret = sum(s,1);
    return ret(0,0)/(double)y.cols();
}

Matrix<double> mse_deriv(const Matrix<double>& y, const Matrix<double>& a)
{
    Matrix<double> ret;
    double m = y.cols();
    ret = (1./(double)m) * (a - y);
    return ret;
}

Matrix<double> read_from_file(const std::string& file, size_t r, size_t c)
{
    ifstream in;
    in.open(file);
    Matrix<double> ret(r,c);
    for(size_t i = 0 ; i < r ; ++i)
    {
        for(size_t j = 0 ; j < c ; ++j)
        {
            double x;
            in >> x;
            ret(i,j) = x;
        }
    }
    return ret;
}

void Model::set_parameters(size_t m)
{
    size_t N = layers.size();
    for(size_t i = 1 ; i < N ; ++i)
    {
        // a = g(z) = w*x + b
        double limit = (6./sqrt(layers[i] + layers[i-1])); //glorot uniform
        Matrix<double> w = real_rand(layers[i], layers[i-1],-limit,limit);
        weights.push_back(w);

        Matrix<double> w_der(w.rows(), w.cols());
        dw.push_back(w_der);

        Matrix<double> b(layers[i],m);
        biases.push_back(b);

        Matrix<double> b_der(b.rows(), b.cols());
        db.push_back(b_der);

        Matrix<double> zi(b.rows(), b.cols());
        zetas.push_back(zi);
        alphas.push_back(zi); 
    }

}


void Model::forward_prop(const Matrix<double> data)
{

    size_t N = weights.size();
    Matrix<double> par = data;
    double (*f)(double);

    for(size_t i = 0 ; i < N ; ++i)
    {
        zetas[i] = multiply(weights[i], par) + biases[i];
        if(activations[i] == "relu")
            f = relu;
        else if(activations[i] == "sigmoid")
            f = sigmoid;
        alphas[i] = apply_function(zetas[i], f);
        par = alphas[i];
    }
}


void Model::backward_prop(const Matrix<double>& data,
                          const Matrix<double>& labels)
{
    size_t N = weights.size();
    size_t m = labels.cols();
    vector< Matrix<double> > da(N);
    vector< Matrix<double> > dz(N);
    double (*g)(double);

    //Matrix<double> da_last = -labels/alphas[alphas.size()-1] + (1.-labels)/(1.-alphas[alphas.size()-1]);
    //Matrix<double> da_last = (1./(double)m) * (alphas[alphas.size()-1] - labels);
    if(loss_function == "cross_entropy")
        da[N-1] = cross_entropy_deriv(labels, alphas[alphas.size()-1]);
    else if(loss_function == "mean_squared_error")
        da[N-1] = mse_deriv(labels, alphas[alphas.size()-1]);
    for(size_t i = (N-1) ; i > 0 ; --i)         
    {
        if(activations[i] == "relu")
            g = relu_deriv;
        else if(activations[i] == "sigmoid")
            g = sigmoid_deriv;
        dz[i] = da[i] * apply_function(zetas[i], g); 
        dw[i] = (1./(double)m) * multiply(dz[i], alphas[i-1].trans());
        db[i] = (1./(double)m) * sum(dz[i],1);
        da[i-1] = multiply(weights[i].trans(), dz[i]); 
    } 
    if(activations[0] == "relu")
        g = relu_deriv;
    else if(activations[0] == "sigmoid")
        g = sigmoid_deriv;
    dz[0] = da[0] * apply_function(zetas[0], g);
    dw[0] = (1./(double)m) * multiply(dz[0], data.trans());
    db[0] = (1./(double)m) * sum(dz[0],1);
    for(size_t i = 0 ; i < db.size() ; ++i) //for every db[i] vector
    {                                       //transforms db from a vector to matrix
        Matrix<double> temp(db[i].rows(), m);
        for(size_t j = 0 ; j < db[i].rows() ; ++j)
        {
            for(size_t k = 0 ; k < m ; ++k)
            {
                temp(j,k) = db[i](j,0);
            }
        }
        db[i] = temp;
    }
}

void Model::gradient_descent(const double& learning_rate) 
{
    size_t N = weights.size();
    for(size_t i = 0 ; i < N ; ++i)
    {
        weights[i] = weights[i] - learning_rate * dw[i]; 
        biases[i] = biases[i] - learning_rate * db[i];
    }
}

void Model::train(const double& learning_rate,
                  size_t batch_size, size_t epochs,
                  const Matrix<double>& train_data,
                  const Matrix<double>& train_labels)
{
    ofstream out;
    out.open("output/loss.dat");


    set_parameters(batch_size);
    size_t nx = train_data.rows();
    size_t  m = train_data.cols();
    size_t N = weights.size();

    size_t Nbatches = m/batch_size;

    for(size_t ep = 0 ; ep < epochs ; ++ep)
    {
        loss = 0.;
        for(size_t i = 0 ; i < Nbatches ; ++i)
        {
            size_t left = batch_size*i;
            size_t right = batch_size*(i+1);

            Matrix<double> batch_data = train_data.sub_matrix(0,nx,left,right);
            Matrix<double> batch_labels = train_labels.sub_matrix(0,0,left,right);

            forward_prop(batch_data);

            double (*f)(const Matrix<double>&, const Matrix<double>&);
            if(loss_function == "mean_squared_error")
                f = mean_squared_error;
            else if(loss_function == "cross_entropy")
                f = cross_entropy;
            loss += f(batch_labels, alphas[alphas.size()-1]);
            //loss += mean_squared_error(batch_labels, alphas[alphas.size()-1]);

            backward_prop(batch_data, batch_labels);

            gradient_descent(learning_rate);
        }
        out << loss/(double)batch_size << endl;
    }
    out.close();
}

void Model::predict(const Matrix<double>& test_data)
{
    size_t m = test_data.cols();
    for(size_t i = 0 ; i < biases.size() ; ++i)
    {
        biases[i] = extend_cols(biases[i], m);
    }

    forward_prop(test_data);
    alphas[alphas.size()-1].print_to_file("output/predictions.dat");

}