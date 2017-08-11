/*****************************************************
   Long Short-Term Memory (LSTM) in C++
   Written by Intelligent Computing Systems Lab (ICSL)
   School of Electrical Engineering
   Yonsei University, Seoul,  South Korea
 *****************************************************/

#include "cell.h"

// LSTM gate
lstm_gate_t::lstm_gate_t() : net(0.0), y(0.0) {}
lstm_gate_t::lstm_gate_t(float m_net, float m_y) : net(m_net), y(m_y) {}
lstm_gate_t::~lstm_gate_t() {}



// LSTM cell state
lstm_state_t::lstm_state_t() :
    previous(0.0), current(0.0), cell(0.0), error(0.0) {}
lstm_state_t::lstm_state_t(float m_previous, float m_current,
                           float, m_cell, float m_error) :  
    previous(m_previous), current(m_current), cell(m_cell), error(m_error) {}
lstm_state_t::~lstm_state_t() {}



// LSTM component
lstm_component_t::lstm_component_t() {}
lstm_component_t::lstm_component_t(float m_
