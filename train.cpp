#include "utils/metrics.hpp"
#include "net/optimizer.hpp"
#include "net/loss.hpp"
#include "net/layer.hpp"
#include "net/functions.hpp"
#include "data/dataset.hpp"
#include "matrix/matrix.hpp"
#include "net/mlp.hpp"

#include <chrono>

//TODO: figure out why test accuracy is 0 (bug)
//TODO: Saving/loading of model
//TODO: Specify model with yaml config? yaml parser
struct elapsTime
{
    std::chrono::steady_clock::time_point start;
    elapsTime() : 
        start( std::chrono::steady_clock::now() ) {}
    double get() { 
        std::chrono::steady_clock::time_point now( std::chrono::steady_clock::now() );
        double et = (std::chrono::duration_cast<std::chrono::microseconds>(now - start).count()) / 1000000.0;
        return et;
    }
};


// "C:\\Users\\hanse\\Documents\\ml_code\\CppNet\\data1000.txt"
std::vector<Matrix<float>> load_dataset(double ptrain, bool shuffle)
{
    std::string filepath;
    std::cout << "Path to dataset: ";
    std::cin >> filepath;

    std::vector<Matrix<float>> data_list;

    // Load & split data
    Dataset data(filepath);
    data.make_split(ptrain, shuffle);

    // Convert to Matrix and retrieve Xs & ys
    auto Xtr = data.toMatrix(DataSplit::TRAIN);
    auto Ytr = data.toMatrix(DataSplit::TRAIN, true); //labels = true
    auto Xdev = data.toMatrix(DataSplit::TEST);
    auto Ydev = data.toMatrix(DataSplit::TEST, true); //labels = true

    data_list.push_back(Xtr); data_list.push_back(Ytr);
    data_list.push_back(Xdev); data_list.push_back(Ydev);
    return data_list;
}

MLP build_model()
{
    // Initialze model with number of input features
    size_t n_inputs;
    std::cout << "Number of input features: ";
    std::cin >> n_inputs;
    std::cout << '\n';

    MLP model(n_inputs);

    // Layer type options
    std::string rel("rel"), lin("lin"), sig("sig");

    // Assemble model layer by layer
    int building;
    std::string layer;
    LayerType lt;
    size_t n; 
    while (building != 0)
    {
        std::cout << "\nLayer type (linear, relu, sigmoid): ";
        std::cin >> layer;
        std::cout << "Number of neurons: ";
        std::cin >> n; 

        // Decipher entered layer type 
        // Convert to lowercase than search for substring
        std::transform(layer.begin(), layer.end(), layer.begin(),
                        [](unsigned char c){ return std::tolower(c); });
        if (layer.find(lin) != std::string::npos)
            { lt = LayerType::DenseLinear; }
        else if (layer.find(rel) != std::string::npos)
            { lt = LayerType::DenseReLU; }
        else 
            { lt = LayerType::DenseSigmoid; }        

        model.add_layer(lt, n);

        std::cout << "Enter 0 to begin training or another character to continue adding layers: ";
        std::cin >> building;
    }

    return model;
}


void train()
{
    Matrix<float> X, Y, Xtest, Ytest;
    auto data_list = load_dataset(0.8, true); //ptrain, shuffle
    X = data_list[0]; //Xtrain
    Y = data_list[1]; //Ytrain
    Xtest = data_list[2];
    Ytest = data_list[3];
    std::cout << "X.shape = "; X.shape();
    std::cout << "Y.shape = "; Y.shape();

    // MLP
    std::default_random_engine gen;
    MLP model = build_model();
    model.initXavier(gen);

    // Optimizer
    OptimSGD sgd;
    sgd.add_model(model);

    // Training Params
    float lr;
    size_t epochs;
    size_t print_every;

    std::cout << "Learning rate: ";
    std::cin >> lr;
    std::cout << "Number of training epochs: ";
    std::cin >> epochs;
    std::cout << "Print updates every _ epochs: ";
    std::cin >> print_every;

    // Training Loop
    BinaryCrossEntropy BCE;
    Matrix<float> pred;

    elapsTime time;
    for (size_t step = 0; step < epochs; step++)
    {
        sgd.zero_grad();
        // Forward
        pred = model.forward(X);
        // Loss
        BCE.compute(pred, Y);
        // Backward
        model.backward(BCE.dInput);
        // Optimize
        sgd.step(lr);        

        // Console Updates
        if (step % print_every == 0)
           std::cout << "epoch " << step 
                     << "[et: " << std::setprecision(2) << time.get() << "s]"
                     << "\t| Loss = " << std::setprecision(5) << BCE.Loss << '\n'; 
    }

    // Post-Training Evalution
    auto round_pred = [](float x) {
        if (x >= 0.5) return 1.0F;
        else return 0.0F;
    };

    // Training Accuracy:
    pred.apply(round_pred);
    auto TR_ACC = metrics::accuracy(pred, Y);
    std::cout << "\nTRAIN ACCURACY = " << TR_ACC;

    // Test accuracy
    Matrix<float> test_pred = model.forward(Xtest);
    auto TE_ACC = metrics::accuracy(test_pred, Ytest);
    std::cout << "\nTEST ACCURACY = " << TE_ACC;
}




int main()
{
    train();
}
