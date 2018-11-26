#include "model.h"

using std::vector;  using std::max;
using std::ifstream;using std::ofstream;
using std::cout;    using std::endl;
using std::string;


/********************************************************************
 * 
 * GENERAL FUNCTIONS
 * 
 *******************************************************************/



/***********************************************
 * Activations 
 ***********************************************/


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


/***********************************************
 * Loss functions 
 ***********************************************/


double cross_entropy(const Matrix<double>& y, const Matrix<double>& a)
{
   Matrix<double> l = -( multiply( y, apply_function(a,log).trans() ) +
                         multiply( (1. - y),
                         apply_function( 1. - a, log ).trans() ));

    return l(0,0);
}


Matrix<double> cross_entropy_deriv(const Matrix<double>& y,
                                   const Matrix<double>& a)
{
    return -y/a + (1.-y)/(1.-a);
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
    double m = y.cols();
    return (1./(double)m) * (a - y);
}


/***********************************************
 * Metrics  
 ***********************************************/


double mean_absolute_error(const Matrix<double>& y, const Matrix<double>& a)
{   //the two matrices must be vectors (1,c) as the output of the network
    size_t c = y.cols();
    Matrix<double> dif = y - a;
    dif = apply_function(dif, fabs);
    Matrix<double> ret = sum(dif,1);
    return ret(0,0)/(double)c; 
}


/***********************************************
 * Initializers  
 ***********************************************/


double glorot_uniform(size_t input_layer, size_t output_layer)
{
    return (6./sqrt(input_layer + output_layer)); 
}


// Reads the input data and returns an array (r,c). The rows and cols
// must be known in advance. Training requires the data to be of the form
// (nx,m)
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



/********************************************************************
 * 
 * MEMBER FUNCTIONS
 * 
 *******************************************************************/


Model::Model(const std::vector<size_t>& l,
             const std::vector<std::string>& act,
             const std::string& name,
             const std::string& opt)
{
    layers = l;
    activations = act;
    loss_function = name;
    optimizer = opt;
    loss = 0.;
}


void Model::set_parameters(size_t m)
{
    size_t N = layers.size();
    for(size_t i = 1 ; i < N ; ++i)
    {
        // a = g(z) = w*x + b

        double limit = glorot_uniform(layers[i-1], layers[i]); 

        Matrix<double> w = real_rand(layers[i], layers[i-1],-limit,limit);
        weights.push_back(w);

        Matrix<double> w_der(w.rows(), w.cols());
        dw.push_back(w_der);

        Matrix<double> opt_w(w.rows(), w.cols(), 1.);
        Sdw.push_back(opt_w);

        Matrix<double> b(layers[i],m);
        biases.push_back(b);

        Matrix<double> b_der(b.rows(), b.cols());
        db.push_back(b_der);

        Matrix<double> opt_b(b.rows(), b.cols(), 1.);
        Sdb.push_back(opt_b);

        Matrix<double> zi(b.rows(), b.cols());
        zetas.push_back(zi);
        alphas.push_back(zi); 
    }

}


void Model::forward_prop(const Matrix<double>& data)
{
    size_t N = weights.size();
    Matrix<double> X = data;
    double (*f)(double);

    for(size_t i = 0 ; i < N ; ++i)
    {
        zetas[i] = multiply(weights[i], X) + biases[i];

        if(activations[i] == "relu")
            f = relu;
        else if(activations[i] == "sigmoid")
            f = sigmoid;
        else
            cout << "Please specify the activation of layer:" << i << endl;

        alphas[i] = apply_function(zetas[i], f);
        X = alphas[i];
    }
}


void Model::backward_prop(const Matrix<double>& data,
                          const Matrix<double>& labels)
{
    size_t N = weights.size();
    size_t m = labels.cols();

    vector< Matrix<double> > da(N);
    vector< Matrix<double> > dz(N);

    double (*g)(double);//function pointer to derivatives of activations 

    if(loss_function == "cross_entropy")
        da[N-1] = cross_entropy_deriv(labels, alphas[alphas.size()-1]);
    else if(loss_function == "mean_squared_error")
        da[N-1] = mse_deriv(labels, alphas[alphas.size()-1]);
    //no need for else. runs after loss calculation in train func.

    for(size_t i = (N-1) ; i > 0 ; --i)         
    {
        if(activations[i] == "relu")
            g = relu_deriv;
        else if(activations[i] == "sigmoid")
            g = sigmoid_deriv;
        //no need for else. runs after forward_prop

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
    
    //db needs broadcasting to have the same dims as b
    for(size_t i = 0 ; i < db.size() ; ++i)
    {
        db[i] = extend_cols(db[i], m);
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


void Model::gradient_descent_with_RMSprop(const double& learning_rate) 
{
    size_t N = weights.size();
    double epsilon = 1.E-8;
    for(size_t i = 0 ; i < N ; ++i)
    {
        weights[i] = weights[i] - learning_rate * dw[i]
                                 /(epsilon + apply_function(Sdw[i], sqrt)); 
        biases[i] = biases[i] - learning_rate * db[i]
                                 /(epsilon + apply_function(Sdb[i], sqrt)); 
    }
}

void Model::train(const double& learning_rate,
                  size_t batch_size, size_t epochs,
                  const Matrix<double>& train_data,
                  const Matrix<double>& train_labels)
{
    ofstream out;
    out.open("output/loss.dat");


    set_parameters(batch_size); //the params could be set up in the cstor
    //but then the biases would need broadcasting here to have cols=batch_size
    size_t nx = train_data.rows();//num of features
    size_t  m = train_data.cols();//num of examples
    size_t  N = weights.size();    //num of parameters

    size_t Nbatches = m/batch_size;
    double beta{0.95}; //parameter of RMSprop

    //function pointer to loss functions
    double (*f)(const Matrix<double>&, const Matrix<double>&);

    if(loss_function == "mean_squared_error")
        f = mean_squared_error;
    else if(loss_function == "cross_entropy")
        f = cross_entropy;
    else
    {
        cout << "Please specify the loss function!" << endl;
        throw "Loss error!\n";
    }

    // function pointer to member functions for parameters update.
    void (Model::*update_func)(const double&);
    if(optimizer == "RMSprop")
        update_func = &Model::gradient_descent_with_RMSprop;
    else
        update_func = &Model::gradient_descent;


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
            loss += f(batch_labels, alphas[alphas.size()-1]);
            backward_prop(batch_data, batch_labels);

            if(optimizer == "RMSprop")
            {
                for(size_t o = 0 ; o < Sdw.size() ; ++o)
                {
                    Sdw[o] = beta * Sdw[o] + (1. - beta) * dw[o] * dw[o];
                    Sdb[o] = beta * Sdb[o] + (1. - beta) * db[o] * db[o];
                }
            }

            (this->*update_func)(learning_rate);
        }
        loss = loss/(double)batch_size;
        out << loss << endl;
    }
    out.close();
}


void Model::predict(const Matrix<double>& new_data)
{
    size_t m = new_data.cols();
    for(size_t i = 0 ; i < biases.size() ; ++i)
    { //biases also need broadcast here for the new data which may have
      // different number of examples
        biases[i] = extend_cols(biases[i], m);
    }

    forward_prop(new_data);
    alphas[alphas.size()-1].print_to_file("output/predictions.dat");

}


double Model::evaluate(const Matrix<double>& new_data,
                     const Matrix<double>& new_labels, const string& metrics)
{
    size_t r = new_data.rows();
    size_t c = new_data.cols();

    predict(new_data);
    Matrix<double> pred = alphas[alphas.size()-1];
    double error = 0.;
    if(metrics == "mae")
    {
        error = mean_absolute_error(new_labels, pred);
    }
    //else if
    // Until the inclusion of other metrics, this is the only one.

    return error;
}