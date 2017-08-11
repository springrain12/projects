/*****************************************************
   Multi-Level Perceptron (MLP) in C++
   Written by Intelligent Computing Systems Lab (ICSL)
   School of Electrical Engineering
   Yonsei University, Seoul,  South Korea
 *****************************************************/

#include <iostream>
#include <libconfig.h++>
#include "mlp.h"


using namespace std;
using namespace libconfig;


mlp_t::mlp_t(string m_config_file_name) :
    config_file_name(m_config_file_name) {}


mlp_t::~mlp_t() {}


void mlp_t::initialize() {
    // User libconfig to parse a configuration file.
    Config mlp_config;
    mlp_config.readFile(config_file_name.c_str());

    try {
        // Read the configuration file.
        num_layers = mlp_config.lookup("num_layers");
        Setting &s_num_neurons_per_layer = mlp_config.lookup("num_neurons_per_layer");
        for(int i = 0; i < s_num_neurons_per_layer.getLength(); i++) {
            num_neurons_per_layer.push_back(s_num_neurons_per_layer[i]);
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
}

