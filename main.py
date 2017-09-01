#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 25 12:13:04 2017

@author: carles
"""

import numpy as np
import imageio
from PIL import Image, ImageFont, ImageDraw
import random

from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers.convolutional import Convolution2D as Conv2D
from keras.layers import MaxPooling2D
from keras.layers import BatchNormalization
from keras.layers import Activation
from keras.layers import Flatten
from keras.layers import Dropout
#from keras.constraints import maxnorm
from keras.optimizers import sgd
#import keras.initializers

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
        
        self.last_frames = []
        [self.remember_frame(self.grid) for i in range(FRAMES_USED)]
        
        
    def remember_frame(self, state):
        self.last_frames.append(state)
        if len(self.last_frames) > FRAMES_USED:
            self.last_frames.pop(0)
            
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
#        return self.grid
        return np.array(self.last_frames) 
        # easier to use built-in arrays and then convert to np.array

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
        self.snake_dir = random.randint(1,4)
        # initialize head and fruit (so that get_free_cell() works)
        self.head_pos =  [-1, -1]
        self.fruit_pos =  [-1, -1]
        # move them to random cells
        self.head_pos = self.get_free_cell()
        self.fruit_pos = self.get_free_cell()
        # paint them
        self.set_tile(self.head_pos, PLAYER) # paint the head
        self.set_tile(self.fruit_pos, FRUIT) # paint the fruit
        
        self.lose_r = -5
        self.survive_r = 1
        self.win_r = 5
    
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
        self.remember_frame(self.grid)
        
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
            self.set_tile(self.head_pos, PLAYER)
            
            # if the fruit was there
            if self.fruit_pos == self.head_pos:
                self.fruit_pos = self.get_free_cell() # reposition fruit
                self.set_tile(self.fruit_pos, FRUIT) # paint fruit
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
        Game.__init__(self, 10, 10)
        self.player_width = 3
        
#        self.player_pos = self.grid_width // 2 - self.player_width // 2
        self.player_pos = random.randint(0, self.grid_width - self.player_width)
        self.fruit_pos = [0, self.grid_width // 2]#random.randint(0, self.grid_width-1)]
            
        
        self.draw_player()
        self.grid[tuple(self.fruit_pos)] = 1
        
        self.extra_info = 0
    
        self.lose_r = -10
        self.survive_r = 0
        self.win_r = 10
        
    def draw_player(self, tile = 1):
        for i in range(self.player_width):
            self.grid[self.grid_height - 1, self.player_pos + i] = tile
            
    def get_actions(self):
        return [0, 1, 2] # left, stay, right
    
    def tile_symbols(self, tile):
        return (" ", "#", "*")[tile]
    
    def transition(self, action):
        reward = self.survive_r
        # Erase player and fruit from the grid
        self.draw_player(0)
        self.grid[tuple(self.fruit_pos)] = 0
        
        # Update fruit
        self.fruit_pos[0] += 1
        
        # Update player
        if action == 0 and self.player_pos != 0:
            self.player_pos -= 1
        if action == 2 and self.player_pos != self.grid_width - self.player_width:
            self.player_pos += 1
            
        # Did the fruit get to the last row?
        if self.fruit_pos[0] == self.grid_height-1:
            
            # Did it catch the fruit?
            if (self.player_pos <= self.fruit_pos[1] and 
                self.fruit_pos[1] <= self.player_pos + self.player_width - 1):
                reward = self.win_r
            else:
                reward = self.lose_r
                self.gameover = True
            # Replace fruit
            self.fruit_pos = [0, random.randint(0, self.grid_width-1)]
            
        self.draw_player()
        self.grid[tuple(self.fruit_pos)] = 1
        
        return reward
    
class Player():
    
    def __init__(self, game):
        # Exploration rate (aka epsilon): determines the probability that the
        # player will take a random action. This avoids the dilemma between
        # exploration and exploitation (should I keep choosing safe actions
        # or do I try to maximize my scores?)
        self.max_epsilon = MAX_EPSILON # 0.1
        self.epsilon = 0.0
        self.epsilon_growth = (self.max_epsilon - self.epsilon
                               )/EPOCHS_TO_MAX_EPSILON #/epoch
        
        # Discount rate: determines how much future rewards are taken into
        # account when training. Zero will make the player myopic (prefer
        # short-term rewards) and one will take the future rewards for
        # exact values (so unless deterministic game make discount < 1)
        self.max_discount = MAX_DISCOUNT #0.9
        self.discount = 0.0
        self.discount_growth = (self.max_discount - self.discount
                                )/EPOCHS_TO_MAX_DISCOUNT #/epoch
        
        # Advantage learning parameter. It is actually k/dt, with dt being
        # the steptime of a frame. Not sure how to set it.
        self.kdt = KDT
        
        self.batch_size = BATCH_SIZE #30
        self.max_mem = MEM_SIZE
        
        self.memory = [] 
        self.game = game
        self.model = self.build_model()
        self.model2 = self.build_model() # for double Q-learning
        
        
        
    def build_model(self):
        # Build deep neural network
        self.input_shape = (self.game.grid_size)
        
        model = Sequential()
        model.add(Dense(DENSE_SIZES[0], activation="relu", 
                             input_shape=self.input_shape,
                             kernel_initializer='random_uniform',
                             bias_initializer='random_uniform'))
        model.add(Dropout(DROPOUT))
        for h in DENSE_SIZES:
            model.add(Dense(h, activation='relu'))
        model.add(Dense(len(self.game.get_actions())))
        
#        self.model.compile("adam", "mse")
        model.compile(sgd(lr=LEARNING_RATE), "mse")
        
        return model
            
##    # Another model with convolutional layers. Comment or uncomment at need.
    def build_model(self):
        
        self.input_shape = (*self.game.grid_shape, FRAMES_USED)
#        fnum = (self.game.grid_height - fshape[0] + 1 
#                )*(self.game.grid_width - fshape[1] + 1)
        model = Sequential()
        for s in CONV_SIZES:
            model.add(BatchNormalization(input_shape=self.input_shape))    
            model.add(Dropout(DROPOUT))
            model.add(Conv2D(s[0], s[1], activation="relu", 
                                 padding='valid',
#                                 subsample=(2, 2), 
#                                 dim_ordering='th',
#                                 input_shape=self.input_shape,
                                 kernel_initializer=INITIALIZER,
                                 bias_initializer=INITIALIZER))
        if POOL_SHAPE != (0, 0):
            model.add(MaxPooling2D(pool_size=POOL_SHAPE))
        model.add(Flatten())
        for s in DENSE_SIZES:
            model.add(BatchNormalization())
            model.add(Dense(s, activation='relu'))
        model.add(Dense(len(self.game.get_actions())))
        model.compile(sgd(lr=LEARNING_RATE), "mse")
        
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
            priority = WIN_PRIORITY
        elif reward == self.game.lose_r:
            priority = LOSE_PRIORITY
        else:
            priority = SUR_PRIORITY
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
    
def train(player, game):
    
    # Training
    print("\n=== Training ===\n")
    
    longest_run = 0
    high_score = 0
    for epoch in range(0, EPOCHS):
        game.reset()
        length = 0
        score = 0 
            
        while not game.gameover and length < 500:
            s = game.get_state() 
            a = player.get_action(s)
            r = game.transition(a)
            sf = game.get_state() 
            
            length += 1
            score += 1 if r == game.win_r else 0
            
#            if length > FRAMES_USED:
            player.memorize(s, a, r, sf, game.gameover)
        loss = player.train()
            
        if VERBOSE_TRAIN:
            print("Epoch {}/{}: \t {} turns. \t Score {}.\t Loss: {:.4f}".format(
                epoch, EPOCHS, length, score, loss))
        longest_run = max(longest_run, length)
        high_score = max(high_score, score)
            
                
    print("Longest run:", longest_run)
    print("Highest score:", high_score)
    
def test(player, game):
        # Testing
    print("\n=== Testing ===\n")
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
                
#        print("Test {}/{}: \t {} turns. \t Score {}.".format(
#                test, tests, length, score))
        longest_run = max(longest_run, length)
        high_score = max(high_score, score)
        
        total_run += length
        total_score += score
                    
    print("Longest run:", longest_run)
    print("Highest score:", high_score)
    print("Average run:", total_run/tests)
    print("Average score:", total_score/tests)    
    
def record_player(player, fname):
    game = player.game
    game.reset()
    length = 0
    score = 0
    tilesize = 24
    imgs = []
    while not game.gameover and length < 1000:
        s = game.get_state()[-1]
        a = player.get_action(s)
        r = game.transition(a)
        
        size = game.grid_height*tilesize, game.grid_width*tilesize
        img = Image.new('RGB', size, 'white')
        draw = ImageDraw.Draw(img)
        for y in range(game.grid_height):
            for x in range(game.grid_width):
                if s[y, x] != 0:
                    box = tilesize*x, tilesize*y, tilesize*(x+1), tilesize*(y+1)
                    draw.rectangle(box, fill="blue")
                    
        
        draw.text((10, 10), "Score: {}".format(score), fill='black',
                  font=ImageFont.truetype("Ubuntu-L.ttf", int(tilesize*0.6)))

        img_array = np.fromstring(img.tobytes(), dtype=np.uint8)
        imgs.append(img_array.reshape((*size, 3)))
        
        length += 1
        score += 1 if r == game.win_r else 0
        
    imageio.mimwrite(fname, imgs)

        

if __name__ == "__main__":
    VERBOSE_TRAIN = True
    
    PLAYER = 1
    FRUIT = 1
    
    BATCH_SIZE = 100
    EPOCHS = 10000
    
    MAX_EPSILON = 0.1
    EPOCHS_TO_MAX_EPSILON = 2000
    MAX_DISCOUNT = 0.95
    EPOCHS_TO_MAX_DISCOUNT = 1000
    KDT = 1.1 # advantage learning parameter k/dt
    
    # Experience replay (prioritized)
    MEM_SIZE = 500
    WIN_PRIORITY = 5
    LOSE_PRIORITY = 5
    SUR_PRIORITY = 1
    
    INITIALIZER = 'random_uniform'
    FRAMES_USED = 4
    CONV_SIZES = ((32, (3, 3)),
                  (64, (3, 3)))
    DENSE_SIZES = (256, 128)
    POOL_SHAPE = (0, 0)
    DROPOUT = 0.25
    LEARNING_RATE = 0.05
    
#     Uncomment this to play manually
#    game = Catch()
#    while not game.gameover:
#        game.draw_screen()
#        print("Choose action:", end="")
#        a = input()
#        r = game.transition(int(a))
#        print("Reward:", r)
#        print("")
#                
    game = Snake()
    player = Player(game)
                 
    train(player, game)
    test(player, game)
    
    record_player(player, 'catchgame.gif')
    
    player.save('snakeplayer')
        
