#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  1 10:18:38 2017

@author: carles
"""
from games import Game
import random

class Snake(Game):
    
    def __init__(self, frames_used):
        Game.__init__(self, 5, 5, frames_used)
        # initialize the snake's direction
        self.snake_dir = random.randint(1,4)
        # initialize head and fruit (so that get_free_cell() works)
        self.head_pos =  [-1, -1]
        self.fruit_pos =  [-1, -1]
        # move them to random cells
        self.head_pos = self.get_free_cell()
        self.fruit_pos = self.get_free_cell()
        # paint them
        self.set_tile(self.head_pos, 1) # paint the head
        self.set_tile(self.fruit_pos, 1) # paint the fruit
        
        self.lose_r = -5
        self.survive_r = -1
        self.win_r = 15
        
    def get_actions(self):
#        return [0, 1, 2, 3, 4] # keep going, move up, right, down, left
        return [0, 1, 2] # keep going, turn left, turn right
    
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
#        # Do nothing if the snake is moving in the opposite direction
#        if action == 1 and self.snake_dir != 3:
#            self.snake_dir = 1
#        if action == 2 and self.snake_dir != 4:
#            self.snake_dir = 2
#        if action == 3 and self.snake_dir != 1:
#            self.snake_dir = 3
#        if action == 4 and self.snake_dir != 2:
#            self.snake_dir = 4
        
        if action == 1: # turn left (counterclockwise)
            self.snake_dir = (0, 4, 1, 2, 3)[self.snake_dir]
        if action == 2: # turn right (clockwise)
            self.snake_dir = (0, 2, 3, 4, 1)[self.snake_dir]
            
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
                self.set_tile(self.fruit_pos, 1) # paint fruit
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