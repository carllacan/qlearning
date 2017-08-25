#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 25 12:13:04 2017

@author: carles
"""


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
        self.grid = []
        for i in range(0, grid_height):
            self.grid.append([])
            for j in range(0, grid_width):
                self.grid[-1].append(0)
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
        Return the grid.
        """
        return self.grid

    def get_actions(self):
        """
        Get the possible actions for this game
        """
        return []
    
    def draw_screen(self):
        """
        Print what is in each cell of the grid.
        """
        for row in self.grid:
            for tile in row:
                print(tile, end="")
            print("")
        
        
class Snake(Game):
    
    def __init__(self):
        Game.__init__(self, 5, 5)
        # initialize head and fruit outside the grid
        self.head_pos =  [-1, -1]
        self.fruit_pos =  [-1, -1]
        self.snake_dir = 0 # which direction is the snake moving
        # move head and fruit somewhere random
        self.head_pos = self.get_free_cell()
        self.fruit_pos = self.get_free_cell()
        self.grid[self.head_pos[0]][self.head_pos[1]] = 1 # occupied by snake
        self.grid[self.fruit_pos[0]][self.fruit_pos[1]] = 2 # occupied by fruit
        
    def get_actions(self):
        return [0, 1, 2, 3, 4] # keep going, move up, right, down, left
    
    
    def transition(self, action):
        """
        Input: Action.
        Change the direction of the snake and then check what happens.
        Mark the game as finished if there is a collision
        Output: reward.
        """
        if action == 1 and self.snake_dir != 3:
            self.snake_dir = 1
        if action == 2 and self.snake_dir != 4:
            self.snake_dir = 2
        if action == 3 and self.snake_dir != 1:
            self.snake_dir = 3
        if action == 4 and self.snake_dir != 2:
            self.snake_dir = 4
            
        if (self.snake_dir == 1 and self.head_pos[0] == 0 or
            self.snake_dir == 2 and self.head_pos[1] == self.grid_width - 1 or
            self.snake_dir == 3 and self.head_pos[0] == self.grid_height - 1 or
            self.snake_dir == 4 and self.head_pos[1] == 0):            
                self.gameover = True
                return -1 # punish the player
        else:
            # remove head
            self.grid[self.head_pos[0]][self.head_pos[1]] = 0
            
            # move head
            if self.snake_dir == 1:
                self.head_pos[0] -= 1
            if self.snake_dir == 2:
                self.head_pos[1] += 1
            if self.snake_dir == 3:
                self.head_pos[0] += 1
            if self.snake_dir == 4:
                self.head_pos[1] -= 1
                
            # replace head (paint over)
            self.grid[self.head_pos[0]][self.head_pos[1]] = 1
            
            # if the fruit was there
            if self.fruit_pos == self.head_pos:
                self.fruit_pos = self.get_free_cell() 
                # lengthen the snake, later on
                self.grid[self.fruit_pos[0]][self.fruit_pos[1]] = 2 # occupied by fruit
                return 5 # reward the player
            else:
                return 0 # no reward for you
            
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