/*****************************************************
   Multi-Level Perceptron (MLP) in C++
   Written by Intelligent Computing Systems Lab (ICSL)
   School of Electrical Engineering
   Yonsei University, Seoul,  South Korea
 *****************************************************/

#include <iostream>
#include <cstdlib>
#include <cstring>
#include "mlp.h"

using namespace std;

int main(int argc, char **argv) {
    // Check # of input arguments.
    if(argc != 3) {
        cout << "Usage: " << argv[0] << " -config <mlp config file>" << endl;
        exit(1);
    }

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

    mlp_t *mlp = new mlp_t(config_file_name);
    mlp->initialize();

    delete mlp;

    return 0;
}