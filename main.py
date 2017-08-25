#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 25 12:13:04 2017

@author: carles
"""

import numpy as np
import random

from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers.convolutional import Convolution2D as Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dropout
from keras.constraints import maxnorm
from keras.optimizers import sgd
import keras.initializers

class Game:
    
    def __init__(self, grid_height, grid_width):
        """
        To make printing easier the coordinates refer to
            distance from the top border
            distance from the left border
        """
        self.grid_height = grid_height
        self.grid_width = grid_width
        self.grid_shape = self.grid_height, self.grid_width
        self.grid_size = grid_height*grid_width
        self.grid = np.zeros(self.grid_shape)
        self.gameover = False
        
        self.extra_info = 0
        
    def transition(self, action):
        """
        Inputs: action
        Each game needs to have a function that computes its next state
        given an action. It also returns the reward.
        Outputs: reward
        """
        return 0 # return reward
    
    def get_state(self):
        """
        Returns the grid.
        """
        return self.grid

    def get_actions(self):
        """
        Get the possible actions for this game
        """
        return []
    
    def tile_symbols(self, tile):
        """
        Prettifies the printed screen. Each game can assign a character to
        each kind of tile. By default use simply the numbers in the grid.
        """
        return tile
    
    def set_tile(self, pos, v):
        """
        Converts whatever pos is to a tuple and modifies the grid
        """
        self.grid[tuple(pos)] = v
        
    
    def draw_screen(self):
        """
        Print what is in each cell of the grid.
        """
        w = len(str(self.tile_symbols(0))) # width of the tile symbols
        print("╔" + "═"*self.grid_width*w + "╗")
        for row in self.grid:
            print("║", end="")
            for tile in row:
                print(self.tile_symbols(int(tile)), end="")
            print("║")
        print("╚" + "═"*self.grid_width*w + "╝")
        
        
    def reset(self):
        self.__init__()
    
        
        
class Snake(Game):
    
    def __init__(self):
        Game.__init__(self, 5, 5)
        # initialize the snake's direction
#        self.snake_dir = 0 
        self.snake_dir = random.randint(1,4)
        # initialize head and fruit (so that get_free_cell() works)
        self.head_pos =  [-1, -1]
        self.fruit_pos =  [-1, -1]
        # move them to random cells
        self.head_pos = self.get_free_cell()
        self.fruit_pos = self.get_free_cell()
        # paint them
        self.set_tile(self.head_pos, 1) # paint the head
        self.set_tile(self.fruit_pos, 2) # paint the head
        
        self.extra_info = 1
        
        self.lose_r = -1
        self.survive_r = 0
        self.win_r = 1
        
    def get_state(self):
        """
        Returns the grid AND the direction of the snake.
        """
        return np.append(self.snake_dir, self.grid)
#        return self.grid
    
    def get_actions(self):
        return [0, 1, 2, 3, 4] # keep going, move up, right, down, left
    
    def tile_symbols(self, tile):
#        return ("░", "▓", "█")[tile]
        return ("   ", " · ", " # ")[tile]
    
    def transition(self, action):
        """
        Input: Action.
        Change the direction of the snake and then check what happens.
        Mark the game as finished if there is a collision
        Output: reward.
        """
        reward = self.survive_r
        # Do nothing if the snake is moving in the opposite direction
        if action == 1 and self.snake_dir != 3:
            self.snake_dir = 1
        if action == 2 and self.snake_dir != 4:
            self.snake_dir = 2
        if action == 3 and self.snake_dir != 1:
            self.snake_dir = 3
        if action == 4 and self.snake_dir != 2:
            self.snake_dir = 4
            
        # Detect collision with borders
        if (self.snake_dir == 1 and self.head_pos[0] == 0 or
            self.snake_dir == 2 and self.head_pos[1] == self.grid_width - 1 or
            self.snake_dir == 3 and self.head_pos[0] == self.grid_height - 1 or
            self.snake_dir == 4 and self.head_pos[1] == 0):            
                self.gameover = True
                reward = self.lose_r # punish the player
        else:
            # remove head 
            self.set_tile(self.head_pos, 0)
            
            # move head
            if self.snake_dir == 1:
                self.head_pos[0] -= 1
            if self.snake_dir == 2:
                self.head_pos[1] += 1
            if self.snake_dir == 3:
                self.head_pos[0] += 1
            if self.snake_dir == 4:
                self.head_pos[1] -= 1
                
            # replace head
            self.set_tile(self.head_pos, 1)
            
            # if the fruit was there
            if self.fruit_pos == self.head_pos:
                self.fruit_pos = self.get_free_cell() # reposition fruit
                self.set_tile(self.fruit_pos, 2) # paint fruit
                reward = self.win_r # reward the player
            
        return reward
                
            
    def get_free_cell(self):
        """
        Returns a random free cell, where there's no snake nor fruit
        """
        free_cells = []
        for y in range(0, self.grid_height):
            for x in range(0, self.grid_width):
                if [y, x] != self.head_pos and [x, y] != self.fruit_pos:
                    free_cells.append([y, x])
        return random.choice(free_cells)
    
    
class Catch(Game):
    
    def __init__(self):
        Game.__init__(self, 3, 3)
        self.player_pos = [self.grid_height - 1, self.grid_width // 2]
        self.fruit_pos = [0, self.grid_width // 2]#random.randint(0, self.grid_width-1)]
            
        self.grid[tuple(self.player_pos)] = 1
        self.grid[tuple(self.fruit_pos)] = 1
        
        self.extra_info = 0
        
        self.lose_r = -1
        self.survive_r = 0
        self.win_r = 1
        
    def get_actions(self):
        return [0, 1, 2] # left, stay, right
    
    def tile_symbols(self, tile):
        return ("   ", "[:]", " * ")[tile]
    
    def transition(self, action):
        reward = self.survive_r
        # Erase player and fruit from the grid
        self.grid[tuple(self.player_pos)] = 0
        self.grid[tuple(self.fruit_pos)] = 0
        
        # Update fruit
        self.fruit_pos[0] += 1
        
        # Update player
        if action == 0 and self.player_pos[1] != 0:
            self.player_pos[1] -= 1
        if action == 2 and self.player_pos[1] != self.grid_width-1:
            self.player_pos[1] += 1
            
        # Did it catch the fruit?
        if self.fruit_pos == self.player_pos:
            self.fruit_pos = [0, random.randint(0, self.grid_width-1)]
            reward = self.win_r
        
        # Did the fruit get off the grid?
        if self.fruit_pos[0] == self.grid_height-1:
            self.fruit_pos = [0, random.randint(0, self.grid_width-1)]
            reward = self.lose_r
            self.gameover = True
            
        self.grid[tuple(self.player_pos)] = 1
        self.grid[tuple(self.fruit_pos)] = 1
        
        return reward
    
class Player():
    
    def __init__(self, game):
        self.epsilon = 0.1
        self.discount_rate = 0.9
        self.max_mem = 500
        self.batch_size = 100
        self.memory = [] 
        self.game = game
        self.build_model()
        
        
    def build_model(self):
        # Build deep neural network
        self.input_shape = (self.game.grid_size + self.game.extra_info,)
        hidden_size = 100
        
        self.model = Sequential()
        self.model.add(Dense(hidden_size, activation="relu", 
                             input_shape=self.input_shape,
                             kernel_initializer='random_uniform',
                             bias_initializer='random_uniform'))
        self.model.add(Dense(hidden_size, activation='relu'))
        self.model.add(Dense(len(game.get_actions()),  activation='relu',
                             kernel_initializer='random_uniform',))
        
        self.model.compile(sgd(lr=.2), "mse")
        
    def shape_grid(self, grid):
        """
        Shapes a grid into an appropriate form for the model
        """
        return grid.reshape(self.input_shape)
    
#    def build_model(self):
#        # Build Convolutional neural network
#        
#        self.input_shape = (*self.game.grid_shape, 1)
#        hidden_size = 100
#        fshape = (2, 2)
#        fnum = (self.game.grid_height - fshape[0] + 1 
#                )*(self.game.grid_width - fshape[1] + 1)
#        
#        self.model = Sequential()
#        self.model.add(Conv2D(fnum, fshape, activation="relu", 
#                             input_shape=self.input_shape))
##        self.model.add(MaxPooling2D(pool_size=(2,2)))
#        self.model.add(Dropout(0.25))
#        self.model.add(Flatten())
#        self.model.add(Dense(hidden_size, activation='relu'))
##        self.model.add(Dense(hidden_size, activation='relu'))
#        self.model.add(Dropout(0.25))
#        self.model.add(Dense(hidden_size, activation='relu'))
#        self.model.add(Dense(len(game.get_actions())))
#        self.model.compile(sgd(lr=.2), "mse")
        
#    def shape_grid(self, state):
#        """
#        Shapes a grid into an appropriate form for the model
#        """
#        return state[1].reshape(self.input_shape)
    
    def forwardpass(self, model_input):
        """
        Input: a SINGLE input for the model
        Wraps a SINGLE input in an array and gets the model's predictions,
        so that we don't have to wrap everytime we want to make a fw pass.
        Output: the predictions for this input
        """
        return self.model.predict(np.array([model_input]))
        
    def get_action(self, state, exploration=True):
        
        if np.random.rand() < self.epsilon and exploration:
            action = random.choice(self.game.get_actions())
        else:
            Q = self.forwardpass(self.shape_grid(self.game.get_state()))[0]
            action = max(self.game.get_actions(), key=lambda a: Q[a])
        return action
            
    def memorize(self, state, action, reward, state_final, gameover):
        self.memory.append((self.shape_grid(state), action, reward, 
                            self.shape_grid(state_final), gameover))
        if len(self.memory) > self.max_mem:
            self.memory.pop(0)
        
    def train(self):   

        n = min(len(self.memory), self.batch_size)
        sample = random.sample(self.memory, n)
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
            targets[i] = self.forwardpass(state_t)[0]
            
            if gameover: 
                # if this action resulted in the end of the game
                # its future reward is just its reward
                targets[i][action_t] = reward_t 
            else:
                # else its future reward is its reward plus the 
                # an approximation of future rewards
                Q = self.forwardpass(state_tp1)[0]
                targets[i][action_t] = reward_t + self.discount_rate*max(Q)
                
        return self.model.train_on_batch(inputs, targets)
                
if __name__ == "__main__":
    
#     Uncomment this to play manually
    game = Catch()
    while not game.gameover:
        game.draw_screen()
        print("Choose action:", end="")
        a = input()
        r = game.transition(int(a))
        print("Reward:", r)
        print("")
                
    game = Catch()
    player = Player(game)
                
    # Training
    
#    print("Training")
    epochs = 200
    longest_run = 0
    high_score = 0
    for epoch in range(0, epochs):
        game.reset()
        length = 0
        score = 0 # compensate for the final -1
        while not game.gameover and length < 50:
            s = game.get_state() 
            a = player.get_action(s)
            r = game.transition(a)
            sf = game.get_state() 
            
#            game.draw_screen()
            
            player.memorize(s, a, r, sf, game.gameover)
            loss = player.train()
            
            length += 1
            score += 1 if r == game.win_r else 0
#        print(length, score, loss)
        print("Epoch {}/{}: \t {} turns. \t Score {}.\t Loss: {:.4f}".format(
                epoch, epochs, length, score, loss))
        longest_run = max(longest_run, length)
        high_score = max(high_score, score)
            
                
print("Longest run:", longest_run)
print("Highest score:", high_score)

# Testing

def test(player, game):
    print("Testing")    
    tests = 100         
    longest_run = 0
    high_score = 0   
    total_run = 0
    total_score = 0
    for test in range(0, tests):
        game.reset()
        length = 0
        score = 0
        while not game.gameover and length < 250:
            s = game.get_state() # get state at the start of the epoch
            a = player.get_action(s, exploration=False)
            r = game.transition(a)
            length += 1
            score += 1 if r == game.win_r  else 0
                
        print("Test {}/{}: \t {} turns. \t Score {}.".format(
                epoch, epochs, length, score))
        longest_run = max(longest_run, length)
        high_score = max(high_score, score)
        
        total_run += length
        total_score += score
                    
    print("Longest run:", longest_run)
    print("Highest score:", high_score)
    print("Average run:", total_run/tests)
    print("Average score:", total_score/tests)
            
test(player, game)
    
