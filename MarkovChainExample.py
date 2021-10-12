#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 30 17:12:30 2021

@author: ubuntu
"""

import numpy as np
from enum import Enum
from functools import partial

class MarkovChain(object):
    def __init__(self, transition_matrix, states):
        """
        Initialize the MarkovChain instance.
 
        Parameters
        ----------
        transition_matrix: 2-D array
            A 2-D array representing the probabilities of change of 
            state in the Markov Chain.
 
        states: 1-D array 
            An array representing the states of the Markov Chain. It
            needs to be in the same order as transition_matrix.
        """
        self.transition_matrix = np.atleast_2d(transition_matrix)
        self.states = states
        self.index_dict = {self.states[index]: index for index in 
                           range(len(self.states))}
        self.state_dict = {index: self.states[index] for index in
                           range(len(self.states))}
 
    def next_state(self, current_state):
        """
        Returns the state of the random variable at the next time 
        instance.
 
        Parameters
        ----------
        current_state: str
            The current state of the system.
        """
        return np.random.choice(
         self.states, 
         p=self.transition_matrix[self.index_dict[current_state], :]
        )
 
    def generate_states(self, current_state, no=10):
        """
        Generates the next states of the system.
 
        Parameters
        ----------
        current_state: str
            The state of the current random variable.
 
        no: int
            The number of future states to generate.
        """
        future_states = []
        for i in range(no):
            next_state = self.next_state(current_state)
            future_states.append(next_state)
            current_state = next_state
        return future_states
class ChannelState(Enum):
    GOOD = 0
    BAD = 1
p = 0.2
r = 0.8
perror = p/(p+r)
initial_state = ChannelState.BAD
states = [ChannelState.GOOD,ChannelState.BAD]
transition_matrix1 = [[1-p,p],[r,1-r]] 
ChannelSim = MarkovChain(transition_matrix1, states)
with open ("str_D1.hevc","rb") as original_stream:
    with open("str_D1_noise.hevc","wb") as noise_stream:
        for chunk in iter(partial(original_stream.read, 1024), b''):
            if (initial_state==ChannelState.GOOD):
                noise_stream.write(chunk)
            initial_state = ChannelSim.next_state(initial_state)