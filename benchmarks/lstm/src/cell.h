/*****************************************************
   Long Short-Term Memory (LSTM) in C++
   Written by Intelligent Computing Systems Lab (ICSL)
   School of Electrical Engineering
   Yonsei University, Seoul,  South Korea
 *****************************************************/

// LSTM gate
class lstm_gate_t {
public:
    lstm_gate_t();
    lstm_gate_t(float m_input, float m_y);
    ~lstm_gate_t();

    float input, y;
};



// LSTM state
class lstm_state_t {
public:
    lstm_state_t();
    lstm_state_t(float m_previous, float m_current, float m_cell, float m_error);
    ~lstm_state_t();

    float previous, current, cell, error;
};



// LSTM component
class lstm_component_t {
public:
    lstm_component_t();
    lstm_component_t(float m_in, float m_forget, float m_out);
    ~lstm_component_t();

    float input, output, forget;
};



// LSTM cell
class lstm_cell_t {
public:
    lstm_cell_t();
    lstm_cell_t(bool m_type);
    ~lstm_cell_t();

    // Input, output, and forget gates
    lstm_gate_t input_gate, output_gate, forget_gate;

    // Cell state
    lstm_state_t state;

    // Weight, delta
    lstm_component_t weight, delta, derivative;

    float gradient, output;
};

