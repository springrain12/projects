/*****************************************************
   Multi-Level Perceptron (MLP) in C++
   Written by Intelligent Computing Systems Lab (ICSL)
   School of Electrical Engineering
   Yonsei University, Seoul,  South Korea
 *****************************************************/

#include <iostream>
#include <cstdlib>
#include <cstring>
#include "mlp_threads.h"

using namespace std;

void print_usage(char *exec) {
    cout << "Usage: " << exec                                   << endl
         << "       -config <required: mlp config file>"        << endl;
    exit(1);
}

int main(int argc, char **argv) {
    // Check # of input arguments.
    if(argc < 3) { print_usage(argv[0]); }

    // Parse input arguments.
    string config_file_name;
    
    for(int i = 1; i < argc; i++) {
        if(!strcasecmp(argv[i],"-config")) {
            config_file_name = argv[++i];
        }
        else {
            cout << "Error: unknown option " << argv[i] << endl;
            exit(1);
        }
    }

    if(!config_file_name.size()) { print_usage(argv[0]); }
    
    mlp_t *mlp = new mlp_t(); 
    
    mlp->initialize(config_file_name); 
    mlp->read_test_img_file();
    mlp->read_test_label_file();
    mlp->read_train_img_file();
    mlp->read_train_label_file();

    mlp->mlp_training();
    mlp->save_weights();
    mlp->mlp_test();

    delete mlp;

    return 0;
}
