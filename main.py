#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 25 12:13:04 2017

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
        self.grid_size = grid_height*grid_width
        self.grid = np.zeros((self.grid_height, self.grid_width))
        self.gameover = False
        
    def transition(self, action):
        """
        Inputs: action
        Each game needs to have a function that computes its next state
        given an action. It also returns the reward.
        Outputs: reward
        """
        return 0 # return reward
    
    def get_screen(self):
        """
        Return the grid. If the game is over return None.
        """
        if self.gameover:
            return None
        else:
            return self.grid.reshape(1, self.grid_width*self.grid_width)

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
        Game.__init__(self, 8, 8)
        # initialize the snake's direction
        self.snake_dir = 0 
        # initialize head and fruit (so that get_free_cell() works)
        self.head_pos =  [-1, -1]
        self.fruit_pos =  [-1, -1]
        # move them to random cells
        self.head_pos = self.get_free_cell()
        self.fruit_pos = self.get_free_cell()
        # paint them
        self.set_tile(self.head_pos, 1) # paint the head
        self.set_tile(self.fruit_pos, 2) # paint the head
        
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
        reward = 0
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
                reward = -1 # punish the player
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
                reward = 5 # reward the player
            
        return reward
                
            
    def get_free_cell(self):
        """
        Returns a random free cell, where there's no snake nor fruit
        """
        free_cells = []
        for y in range(0, self.grid_height):
            for x in range(0, self.grid_width):
                if [x, y] != self.head_pos and [x, y] != self.fruit_pos:
                    free_cells.append([x, y])
        return random.choice(free_cells)
    
if __name__ == "__main__":
    
    game = Snake()
    while not game.gameover:
        game.draw_screen()
        print("Choose action:", end="")
        a = input()
        r = game.transition(int(a))
        print("Reward:", r)
        print("")