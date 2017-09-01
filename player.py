#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  1 10:21:48 2017

@author: carles
"""
import random
import numpy as np

from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers.convolutional import Convolution2D as Conv2D
from keras.layers import MaxPooling2D
from keras.layers import BatchNormalization
from keras.layers import Flatten
from keras.layers import Dropout
#from keras.constraints import maxnorm
from keras.optimizers import sgd
#import keras.initializers

class Player():
    
    def __init__(self, game, 
                 max_epsilon, 
                 epochs_to_max_epsilon,
                 max_discount,
                 epochs_to_max_discount,
                 kdt,
                 batch_size,
                 mem_size,
                 win_priority,
                 lose_priority,
                 sur_priority,
                 kernel_initializer,
                 bias_initializer,
                 frames_used,
                 convolutional_sizes,
                 dense_sizes,
                 pool_shape,
                 dropout,
                 learning_rate
                 ):
        
        # Exploration rate (aka epsilon): determines the probability that the
        # player will take a random action. This avoids the dilemma between
        # exploration and exploitation (should I keep choosing safe actions
        # or do I try to maximize my scores?)
        self.max_epsilon = max_epsilon # 0.1
        self.epsilon = 0.0
        self.epsilon_growth = (self.max_epsilon - self.epsilon
                               )/epochs_to_max_epsilon #/epoch
        
        # Discount rate: determines how much future rewards are taken into
        # account when training. Zero will make the player myopic (prefer
        # short-term rewards) and one will take the future rewards for
        # exact values (so unless deterministic game make discount < 1)
        self.max_discount = max_discount #0.9
        self.discount = 0.0
        self.discount_growth = (self.max_discount - self.discount
                                )/epochs_to_max_discount #/epoch
        
        # Advantage learning parameter. It is actually k/dt, with dt being
        # the steptime of a frame. Not sure how to set it.
        self.kdt = kdt
        
        self.batch_size = batch_size 
        self.max_mem = mem_size
        
        self.win_priority = win_priority
        self.lose_priority = lose_priority
        self.sur_priority = sur_priority
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.frames_used = frames_used
        self.convolutional_sizes = convolutional_sizes
        self.dense_sizes = dense_sizes
        self.pool_shape = pool_shape
        self.dropout = dropout
        self.learning_rate = learning_rate
        
        self.memory = [] 
        self.game = game
        self.model = self.build_model()
        self.model2 = self.build_model() # for double Q-learning
        
        
        
    def build_model(self):
        # Build deep neural network
        self.input_shape = (self.game.grid_size)
        
        model = Sequential()
        model.add(Dense(self.dense_sizes[0], activation="relu", 
                             input_shape=self.input_shape,
                             kernel_initializer='random_uniform',
                             bias_initializer='random_uniform'))
        model.add(Dropout(self.dropout))
        for h in self.dense_sizes:
            model.add(Dense(h, activation='relu'))
        model.add(Dense(len(self.game.get_actions())))
        
#        self.model.compile("adam", "mse")
        model.compile(sgd(lr=self.learning_rate), "mse")
        
        return model
            
##    # Another model with convolutional layers. Comment or uncomment at need.
    def build_model(self):
        
        self.input_shape = (*self.game.grid_shape, self.frames_used)
#        fnum = (self.game.grid_height - fshape[0] + 1 
#                )*(self.game.grid_width - fshape[1] + 1)
        model = Sequential()
        for s in self.convolutional_sizes:
            model.add(BatchNormalization(input_shape=self.input_shape))    
            model.add(Dropout(self.dropout))
            model.add(Conv2D(s[0], s[1], activation="relu", 
                                 padding='valid',
#                                 subsample=(2, 2), 
#                                 dim_ordering='th',
#                                 input_shape=self.input_shape,
                                 kernel_initializer=self.kernel_initializer,
                                 bias_initializer=self.bias_initializer))
        if self.pool_shape != (0, 0):
            model.add(MaxPooling2D(pool_size=self.pool_shape))
        model.add(Flatten())
        for s in self.dense_sizes:
            model.add(BatchNormalization())
            model.add(Dense(s, activation='relu'))
        model.add(Dense(len(self.game.get_actions())))
        model.compile(sgd(lr=self.learning_rate), "mse")
        
        return model
    
        
    def shape_grid(self, state):
        """
        Shapes a grid into an appropriate form for the model
        """
        return state.reshape(self.input_shape)
    
    def forwardpass(self, model_input, secondary_model = False):
        """
        Input: a SINGLE input for the model
        Wraps a SINGLE input in an array and gets the model's predictions,
        so that we don't have to wrap everytime we want to make a fw pass.
        Output: the predictions for this input
        """
        model = self.model if not secondary_model else self.model2
        return model.predict(np.array([model_input]))
        
    def get_action(self, state, exploration=True):
        
        if random.random() < self.epsilon and exploration:
            action = random.choice(self.game.get_actions())
        else:
            Q = self.forwardpass(self.shape_grid(self.game.get_state()))[0]
#            action = max(self.game.get_actions(), key=lambda a: Q[a])
            A = max(Q) + (Q - max(Q))*self.kdt 
            action = max(self.game.get_actions(), key=lambda a: A[a])
        return action
            
    def memorize(self, state, action, reward, state_final, gameover):
        experience = (self.shape_grid(state), action, reward, 
                            self.shape_grid(state_final), gameover)
        # Prioritized Experience Replay
        if reward == self.game.win_r:
            priority = self.win_priority
        elif reward == self.game.lose_r:
            priority = self.lose_priority
        else:
            priority = self.sur_priority
        for i in range(priority):
            self.memory.append(experience)
            if len(self.memory) > self.max_mem:
                self.memory.pop(0)
        
    def train(self):   

        # grow the discount rate
        self.discount += self.discount_growth
        self.discount = min(self.max_discount, self.discount) # bound
        
        # grow the exploration rate
        self.epsilon += self.epsilon_growth
        self.epsilon = min(self.max_epsilon, self.epsilon) # bound
        
        n = min(len(self.memory), self.batch_size)
        
        # Take the sample at random
        sample = random.sample(self.memory, n)
        # Take the last experiences, last in first out
#        sample = reversed(self.memory[-n:])
        
        # Take the sample at random but prioritizing the last experiences
#        w = np.linspace(0.1, 1, num = len(self.memory))
#        w = w/np.sum(w)
#        s = np.random.choice(len(self.memory), p=w, size=n, replace=False)
#        sample = np.array(self.memory)[s]
        
        inputs = np.zeros((n, *(self.input_shape)))
        targets = np.zeros((n, len(self.game.get_actions())))
        
        for i, experience in enumerate(sample):
            state_t = experience[0]
            action_t = experience[1]
            reward_t = experience[2]
            state_tp1 = experience[3]
            gameover = experience[4]
            
            # make the input vector
            inputs[i] = state_t
            
            # make the target vector
            targets[i] = self.forwardpass(state_t, True)[0]
            
            if gameover: 
                # if this action resulted in the end of the game
                # its future reward is just its reward
                targets[i][action_t] = reward_t 
            else:
                # else its future reward is its reward plus the 
                # an approximation of future rewards
                Q = self.forwardpass(state_tp1, True)[0]
#                targets[i][action_t] = reward_t + self.discount*max(Q)
                nextQ = Q[self.get_action(state_tp1)]
                targets[i][action_t] = reward_t + self.discount*nextQ #SARSA
                
        # Update secondary network weights to those of the primary
        self.model2.set_weights(self.model.get_weights())
        return self.model.train_on_batch(inputs, targets)
    
    def save(self, fname):
        self.model.save_weights(fname + '.h5')
        
    def load(self, fname):
        self.model.save_weights(fname + '.h5')