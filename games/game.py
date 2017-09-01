#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  1 10:18:38 2017

@author: carles
"""

import numpy as np
import random

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