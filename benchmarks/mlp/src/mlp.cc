/*****************************************************
   Multi-Level Perceptron (MLP) in C++
   Written by Intelligent Computing Systems Lab (ICSL)
   School of Electrical Engineering
   Yonsei University, Seoul,  South Korea
 *****************************************************/

#include <iostream>
#include <libconfig.h++>
#include <random>
#include "mlp.h"


using namespace std;
using namespace libconfig;


mlp_t::mlp_t(string m_config_file_name,
             string m_weight_file_name) :
    config_file_name(m_config_file_name),
    weight_file_name(m_weight_file_name) {
}


mlp_t::~mlp_t() {
    for(unsigned i = 0; i < num_layers-1; i++) {
        for(unsigned j = 0; j < num_neurons_per_layer[i+1]; j++) {
            delete [] weights[i][j];
        }
        delete [] weights[i];
    }
    delete [] weights;

    num_neurons_per_layer.clear();
}


void mlp_t::initialize() {
    // User libconfig to parse a configuration file.
    Config mlp_config;
    mlp_config.readFile(config_file_name.c_str());

    try {
        // Read the configuration file.
        num_layers  = mlp_config.lookup("num_layers");
        Setting &s_num_neurons_per_layer = mlp_config.lookup("num_neurons_per_layer");
        for(int i = 0; i < s_num_neurons_per_layer.getLength(); i++) {
            num_neurons_per_layer.push_back(s_num_neurons_per_layer[i]);
        }

        // Vector of weight matrix. "-1" means there are
        // no more weights from the output layer.
        weights = new float**[num_layers-1];
        for(unsigned i = 0; i < num_layers-1; i++) {
            weights[i] = new float*[num_neurons_per_layer[i+1]];
            for(unsigned j = 0; j < num_neurons_per_layer[i+1]; j++) {
                weights[i][j] = new float[num_neurons_per_layer[i]+1];
            }
        }

        if(!weight_file_name.size()) { init_weights(); }
        else { load_weights(); }

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



void mlp_t::init_weights() {
    default_random_engine generator;
    normal_distribution<float> distribution(0.0, 0.01);

    for(unsigned i = 0; i < num_layers-1; i++) {
        for(unsigned j = 0; j < num_neurons_per_layer[i+1]; j++) {
            for(unsigned k = 0; k < num_neurons_per_layer[i]; k++) {
                weights[i][j][k] = distribution(generator);
            }
        }
    }
}

void mlp_t::load_weights() {
}

