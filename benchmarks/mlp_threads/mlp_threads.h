/*****************************************************
   Multi-Level Perceptron (MLP) in C++
   Written by Intelligent Computing Systems Lab (ICSL)
   School of Electrical Engineering
   Yonsei University, Seoul,  South Korea
 *****************************************************/

#include <stdint.h>
#include <string>
#include <vector>
#include <mutex>
#include <thread>


// MLP layer types
enum MLP_LAYERS { NONE = 0, CONV, POOL, CLASS, NUM_LAYER_TYPES };

// MLP class
class mlp_t {
public:
    mlp_t();                                             // MLP constructor
    virtual ~mlp_t();                                    // MLP destructor
   
    void initialize(std::string m_config_file_name);     // Initialize MLP parameters
    void read_test_img_file();
    void read_test_label_file();
    void read_train_img_file();
    void read_train_label_file();
    void init_weights();
    void load_weights();
    void save_weights();
    void forward_propagation(float **);
    void mlp_training();
    void mlp_test();
    void mlp_calculation_per_thread(unsigned);
    void mlp_train_per_thread(unsigned);
    void softmax(float **); 
    unsigned find_max(float **);
    float sigmoid(float x);
private:
    float **neurons;
    bool require_training;
    float **weights;
    float **delta;
    float **theta;

    std::string config_file_name;                        // Configuration file name
    std::string test_img_file_name;                      // Test img file for inferencing
    std::string test_label_file_name;                    // Test label file for inferencing
    std::string train_img_file_name;                     // MLP train img file name
    std::string train_label_file_name;                   // MLP train label file name 
    std::string weight_file_name;                        // Pre-trained weight file name 
    std::mutex my_mutex;
    
    unsigned num_layers;
    unsigned total_layer_index;
    unsigned num_threads;
    unsigned per_thread_index;
    unsigned correct_count;

	unsigned num_neurons_in_input_layer;
    unsigned num_neurons_in_hidden_layer;
    unsigned num_neurons_in_output_layer;
    unsigned *num_neurons_per_layer;
   
    unsigned test_set_size;
    unsigned train_set_size;
    uint8_t *test_img_set;
    uint8_t *train_img_set;
    uint8_t *test_label_set;
    uint8_t *train_label_set;
	float *answer_set;

    float learning_rate;
    float momentum;
};