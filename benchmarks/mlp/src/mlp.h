/*****************************************************
   Multi-Level Perceptron (MLP) in C++
   Written by Intelligent Computing Systems Lab (ICSL)
   School of Electrical Engineering
   Yonsei University, Seoul,  South Korea
 *****************************************************/

#include <string>
#include <vector>

// MLP layer types
enum MLP_LAYERS { NONE = 0, CONV, POOL, CLASS, NUM_LAYER_TYPES };

// MLP class
class mlp_t {
public:
    mlp_t(std::string m_config_file_name,
          std::string m_weight_file_name);      // MLP constructor
    virtual ~mlp_t();                           // MLP destructor

    void initialize();                          // Initialize MLP parameters

private:
    std::string config_file_name;               // Configuration file name
    std::string weight_file_name;               // Pre-trained weight file name
    unsigned num_layers;                        // # of layers including I/O
    std::vector<unsigned> num_neurons_per_layer;// # of neurons in each layer
    float ***weights;                           // Vector of 2D weight matrix

    void init_weights();
    void load_weights();

    /*
    float *output;
    int *num_units;
    float **units;
    float *needed_output;
    int *bias;
    float learning_rate;
    float **delta;
    */
};

