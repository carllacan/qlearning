#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  1 10:18:38 2017

@author: carles
"""

from game import Game
import random

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