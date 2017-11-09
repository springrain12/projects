/*****************************************************
   Multi-Level Perceptron (MLP) in C++
   Written by Intelligent Computing Systems Lab (ICSL)
   School of Electrical Engineering
   Yonsei University, Seoul,  South Korea
 *****************************************************/

#include <fstream>
#include <iostream>
#include <libconfig.h++>
#include <random>
#include <algorithm>
#include <cmath>
#include "mlp_threads.h"

using namespace std;
using namespace libconfig;

mlp_t::mlp_t() :
    require_training(false),
    test_img_set(NULL),
    train_img_set(NULL),
    test_label_set(NULL),
	train_label_set(NULL) {
}

mlp_t::~mlp_t() {
    if(weights) {
        for(unsigned i = 0; i < total_layer_index; i++){
            delete [] weights[i];
        }
    }
    delete [] weights;
    delete [] test_img_set;
    delete [] test_label_set;
    delete [] train_img_set;
    delete [] train_label_set;
    delete [] num_neurons_per_layer;
}

void mlp_t::initialize(string m_config_file_name) {

    // User libconfig to parse a configuration file.
    config_file_name = m_config_file_name;
    Config mlp_config;
    mlp_config.readFile(config_file_name.c_str());

    try {
        test_img_file_name = mlp_config.lookup("test_img").c_str();
        test_label_file_name = mlp_config.lookup("test_label").c_str();
        if(mlp_config.exists("weight")) {
             weight_file_name = mlp_config.lookup("weight").c_str();
            if(mlp_config.exists("train_img") ||
               mlp_config.exists("train_label")) {
                cout << "Warning: train_img and train_label will be ignored" << endl;
            }
            require_training = false;
        }
        else {
            train_img_file_name = mlp_config.lookup("train_img").c_str();
            train_label_file_name = mlp_config.lookup("train_label").c_str();
            if(mlp_config.exists("weight")) {
                cout << "Warning: pre-trained weight will be ignored" << endl;
            }
            require_training = true;
        }
            
        // Load hidden layer and image size array settings.
        Setting &s_num_neurons_in_hidden_layer = mlp_config.lookup("num_neurons_in_hidden_layer");
        Setting &s_image_size = mlp_config.lookup("image_size");
        Setting &s_num_neurons_in_output_layer = mlp_config.lookup("num_neurons_in_output_layer");


        // +1 for input and another +1 for output layer.
        num_layers = s_num_neurons_in_hidden_layer.getLength()+2;
		total_layer_index = num_layers-1;

        // Load size of input images.
        if(s_image_size.getLength() != 2) {
            cerr << "image_size must be defined [width, length]" << endl;
            exit(1);
        }
       
        // Setting the number of each layer.
        num_neurons_per_layer = new unsigned[num_layers];
       
        // Load the # of neurons in input layer.
        num_neurons_in_input_layer = unsigned(s_image_size[0]) * unsigned(s_image_size[1]); // Image width * Image length
        num_neurons_per_layer[0] = num_neurons_in_input_layer;
        
        // Load the number of neurons in each hidden layer.
        for(int i = 1; i <= s_num_neurons_in_hidden_layer.getLength(); i++) {
            num_neurons_per_layer[i] = unsigned(s_num_neurons_in_hidden_layer[i-1]);
        }

        // Load the # of neurons in output layer.
        num_neurons_per_layer[total_layer_index] = unsigned(s_num_neurons_in_output_layer);
        num_neurons_in_hidden_layer = num_neurons_per_layer[1];
        num_neurons_in_output_layer = num_neurons_per_layer[2]; 
		
        // Setting weights
        weights = new float*[total_layer_index];
        for(unsigned i = 0; i < total_layer_index; i++) {
            weights[i] = new float[num_neurons_per_layer[i] * num_neurons_per_layer[i+1]];
        }
        
        // Load the number of training and test set.
        test_set_size = unsigned(mlp_config.lookup("test_set_size"));
        train_set_size = unsigned(mlp_config.lookup("train_set_size"));
        
        // Load the learning rate.
        learning_rate = float(mlp_config.lookup("learning_rate"));
        momentum = float(mlp_config.lookup("momentum"));
         
        // Setting test label and img set.
        test_label_set = new uint8_t[test_set_size];
        test_img_set = new uint8_t[test_set_size * num_neurons_in_input_layer];

        // Setting train label and img set.
        train_label_set = new uint8_t[train_set_size];
        train_img_set = new uint8_t[train_set_size * num_neurons_in_input_layer];

        // Load number of threads
        num_threads = mlp_config.lookup("num_threads");
   
        if(!weight_file_name.size()) { init_weights();
        cout << "init_weights" << endl;
        }
        else{ load_weights(); 
        cout << "load_weights" << endl;
        }
    }
    catch(SettingNotFoundException e) {
        cout << "Error: " << e.getPath() << " is not defined in "
             << config_file_name << endl;
    }
    catch(SettingTypeException e) {
        cout << "Error: " << e.getPath() << " has incorrect type in "
         << config_file_name << endl;
    }
    catch(FileIOException e) {
        cout << "Error: " << config_file_name << " does not exist" << endl;
    }
    catch(ParseException e) {
        cout << "Error: Failed to parse line # " << e.getLine()
             << " in " << config_file_name << endl;
    }
}

// Initialize weights
void mlp_t::init_weights() {
    default_random_engine generator;
    normal_distribution <float> distribution(0.0, 0.01); // Mean = 0.0, Variance = 0.01
    for(unsigned i = 0; i < total_layer_index; i++){
        for(unsigned j = 0; j < num_neurons_per_layer[i] * num_neurons_per_layer[i+1]; j++) {
            weights[i][j] = distribution(generator);
        }
    }
}

// Load weights from pre-trained weight file
void mlp_t::load_weights() {
    fstream file_stream;
    file_stream.open(weight_file_name.c_str(), fstream::in);

    if(!file_stream.is_open()) {
        cerr << "Error: failed to open" << weight_file_name << endl;
        exit(1);
    }
    for(unsigned i = 0; i < total_layer_index; i++) {
        for(unsigned j = 0; j < num_neurons_per_layer[i] * num_neurons_per_layer[i+1]; j++){
            file_stream >> weights[i][j];
        }
    }
}

void mlp_t::save_weights() {
    
    if(require_training) {
        fstream file_stream("my_weights.txt", ios::out);
        for(unsigned i = 0; i < total_layer_index; i++) {
            for(unsigned j = 0; j < num_neurons_per_layer[i] * num_neurons_per_layer[i+1]; j++) {
                file_stream << weights[i][j] << endl;
            }
        }
        file_stream.close();
    }
    else return;
}

// Read test image file
void mlp_t::read_test_img_file() {
    fstream file_stream;
    file_stream.open(test_img_file_name.c_str(), fstream::in|fstream::binary);
 
    if(!file_stream.is_open()){
        cerr << "Error: failed to open " << test_img_file_name << endl;
        exit(1);
    }
    // Read magic number
    int magic_number;
    for(unsigned i = 0; i < 4; i++) {
        file_stream.read((char*)&magic_number,sizeof(int));
    }   
    // Read test image
    uint8_t img;
    for(unsigned i = 0; i < test_set_size * num_neurons_in_input_layer ; i++) { 
        file_stream.read((char*)&img,sizeof(uint8_t));
        test_img_set[i] = img;
    }
}

// Read test label file
void mlp_t::read_test_label_file() {
    fstream file_stream;
    file_stream.open(test_label_file_name.c_str(), fstream::in|fstream::binary);

    if(!file_stream.is_open()){
        cerr<< "Error: failed to open" << test_label_file_name << endl;
        exit(1);
    }
    // Read magic number
    int magic_number;
    for(unsigned i = 0; i < 2; i++) {
        file_stream.read((char*)&magic_number,sizeof(int));
    }
    // Read test label
    uint8_t label;
    for(unsigned i = 0; i < test_set_size; i++) {
        file_stream.read((char*)&label,sizeof(uint8_t));
        test_label_set[i] = label;
    }
}

// Read train image file
void mlp_t::read_train_img_file(){
    if(!require_training) return;

    fstream file_stream;
    file_stream.open(train_img_file_name.c_str(), fstream::in|fstream::binary);

    if(!file_stream.is_open()){
        cerr << "Error: failed to open" << train_img_file_name << endl;
        exit(1);
    }
    // Read magic number
    int magic_number;
    for(unsigned i = 0 ; i < 4; i++) {
        file_stream.read((char*)&magic_number,sizeof(int));
    }
    // Read train image
    uint8_t img;
    for(unsigned i = 0 ; i < train_set_size * num_neurons_in_input_layer ; i++) {
        file_stream.read((char*)&img,sizeof(uint8_t));
        train_img_set[i] = img;
    }
}

// Read train label file
void mlp_t::read_train_label_file(){
    if(!require_training) return;

    fstream file_stream;
    file_stream.open(train_label_file_name.c_str(), fstream::in|fstream::binary);

    if(!file_stream.is_open()){
        cerr<< "Error: failed to open" << train_label_file_name << endl;
        exit(1);
    }
    // Read magic number
    int magic_number;
    for(unsigned i = 0 ; i < 2; i++) {
        file_stream.read((char*)&magic_number,sizeof(int));
    }
    // Read train label
    uint8_t label;
    for(unsigned i = 0 ; i < train_set_size ; i++) {
        file_stream.read((char*)&label,sizeof(uint8_t));
        train_label_set[i] = label;
    }
}
   

void mlp_t::mlp_training() {
    if(require_training) {
        cout << "start training" << endl;

        // Spawn threads
        thread t[num_threads];    
        for(unsigned i = 0; i < num_threads; i++) {
            t[i] = thread(&mlp_t::mlp_train_per_thread, this, i);
        }
        for(unsigned i = 0; i < num_threads; i++) {
            t[i].join();
        }

    }
                    
    else return;
}

void mlp_t::mlp_test() {
    
    correct_count = 0;

    cout << "start test" << endl;    
    
    // Spawn threads
    thread t[num_threads];
    for(unsigned i = 0; i < num_threads; i++) {
        t[i] = thread(&mlp_t::mlp_test_per_thread, this, i);
    }
    
    for(unsigned i = 0; i < num_threads; i++) {
        t[i].join();
    }
    
    cout << endl << "correct ratio is : " << correct_count << " / " << test_set_size << endl;
}

void mlp_t::mlp_test_per_thread(unsigned tid) {
    
    cout << "thread id: " << tid << endl;
  
    float **tmp_neurons;
    tmp_neurons = new float*[num_layers];
    for(unsigned i = 0; i < num_layers; i++) {
	    tmp_neurons[i] = new float[num_neurons_per_layer[i]];
    }
    
    for(unsigned i = tid * (test_set_size + num_threads - 1) / num_threads; i < (tid+1) * (test_set_size + num_threads - 1) / num_threads; i++) { 
        for(unsigned j = 0; j < num_neurons_in_input_layer; j++) {
            tmp_neurons[0][j] = float(test_img_set[i * num_neurons_in_input_layer + j]);
        }
    
    forward_propagation(tmp_neurons);
    softmax(tmp_neurons);
     
    unsigned output = find_max(tmp_neurons);
   
    cout <<"test label set # " << i+1 << "    " << "test label is: " << (unsigned)test_label_set[i] 
         <<"    "<< "test answer is: "<< output << endl;
        
        if(unsigned(test_label_set[i]) == output) {
            std::lock_guard<std::mutex> guard(my_mutex);
            correct_count += 1;
        }
    }

    if(tmp_neurons) {
        for(unsigned i = 0; i < num_layers; i++) {
            delete [] tmp_neurons[i];
        }
    }
    delete [] tmp_neurons;
}

void mlp_t::mlp_train_per_thread(unsigned tid) {
     
    // For train
    if(require_training) {
    
        float **tmp_neurons;
        tmp_neurons = new float*[num_layers];
        for(unsigned i = 0; i < num_layers; i++) {
	        tmp_neurons[i] = new float[num_neurons_per_layer[i]];
        }
        float *tmp_answer_set;
        tmp_answer_set = new float[num_neurons_in_output_layer];
        float **tmp_theta;
        tmp_theta = new float*[total_layer_index];
        for(unsigned i = 0; i < total_layer_index; i++) {
            tmp_theta[i] = new float[num_neurons_per_layer[i+1]];
        }
        float **tmp_delta;
        tmp_delta = new float*[total_layer_index];
        for(unsigned i = 0; i < total_layer_index; i++) {
            tmp_delta[i] = new float[num_neurons_per_layer[i] * num_neurons_per_layer[i+1]];
        }
        
        float sum; 
        for(unsigned i = tid * (train_set_size + num_threads - 1) / num_threads; i < (tid+1) * (train_set_size + num_threads - 1) / num_threads; i++) {
            // Setting training answers
            for(unsigned j = 0; j < num_neurons_in_output_layer; j++) {
                if(j == (unsigned)train_label_set[i]) { tmp_answer_set[j] = 1.0; }
                else { tmp_answer_set[j] = 0.0;}
            }
            
            // Setting training images into neurons @ input layer
            for(unsigned j = 0; j < num_neurons_in_input_layer; j++) {
                tmp_neurons[0][j] = float(train_img_set[i * num_neurons_in_input_layer + j]);
            }
            forward_propagation(tmp_neurons);
            
            // Calculate theta
            for(unsigned j = 0; j < num_neurons_in_output_layer; j++) {
                tmp_theta[total_layer_index-1][j] = tmp_neurons[total_layer_index][j] * (1.0 - tmp_neurons[total_layer_index][j]) * (tmp_answer_set[j] - tmp_neurons[total_layer_index][j]);
            }   
            for(unsigned j = total_layer_index - 1; j > 0; j--) {
                for(unsigned k = 0; k < num_neurons_per_layer[j]; k++) {
                    sum = 0.0;
                    for(unsigned l = 0; l< num_neurons_per_layer[j+1]; l++) {
                        sum += tmp_theta[j][l] * weights[j][l * num_neurons_per_layer[j] + k];
                    }
                    tmp_theta[j-1][k] = tmp_neurons[j][k] * (1.0 - tmp_neurons[j][k]) * sum;
                }
            }
            
            for(unsigned j = 0; j < total_layer_index; j++) {
                for(unsigned k = 0; k < num_neurons_per_layer[j+1]; k++) {
                    for(unsigned l = 0; l < num_neurons_per_layer[j]; l++) {
                        tmp_delta[j][k * num_neurons_per_layer[j] + l] = learning_rate * tmp_theta[j][k] * tmp_neurons[j][l];
                        //std::lock_guard<std::mutex> guard(my_mutex);
                        weights[j][k * num_neurons_per_layer[j] + l] += tmp_delta[j][k * num_neurons_per_layer[j] + l] + momentum * tmp_delta[j][k * num_neurons_per_layer[j] + l];
                    }
                }
            }      
            cout << "train image set # " << i+1 << "    "
                 << "train label is: " << (unsigned) train_label_set[i] << endl;
        }   
        if(tmp_neurons) {
            for(unsigned i = 0; i < num_layers; i++) {
            delete [] tmp_neurons[i];
            }
        }
        if(tmp_delta) {
            for(unsigned i = 0; i < total_layer_index; i++) {
                delete [] tmp_delta[i];
            }
        }
        if(tmp_theta) {
            for(unsigned i = 0; i < total_layer_index; i++) {
                delete [] tmp_theta[i];
            }
        }
        delete [] tmp_neurons;
        delete [] tmp_delta;
        delete [] tmp_theta;
    
    }
    else return;
}

void mlp_t::forward_propagation(float **neurons) {
    for(unsigned i = 0; i < total_layer_index; i++) {
        float sum;
        for(unsigned j = 0; j < num_neurons_per_layer[i+1]; j++) {
            sum = 0.0;
            for(unsigned k = 0; k < num_neurons_per_layer[i]; k++) {
                sum += weights[i][j * num_neurons_per_layer[i] + k] * neurons[i][k];
            }
            neurons[i+1][j] = sigmoid(sum);
        }
    }
}   

void mlp_t::softmax(float **neurons) {
    float sum = 0.0;
    for(unsigned i = 0; i < num_neurons_in_output_layer; i++) {
        sum += exp(neurons[total_layer_index][i]);
    }
    for(unsigned i = 0; i < num_neurons_in_output_layer; i++) {
        neurons[total_layer_index][i] = exp(neurons[total_layer_index][i]) / sum;
    }
}

unsigned mlp_t::find_max(float **neurons) {
    float max_val = 0.0;
    unsigned index = 0;
    for(unsigned i = 0; i < num_neurons_in_output_layer; i++) {
        if(max_val < neurons[total_layer_index][i]) {
            max_val = neurons[total_layer_index][i];
            index = i;
        }
    }
    return index;
}

float mlp_t::sigmoid(float x) {
    return 1 / (1 + exp(-x)); }
