#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 25 12:13:04 2017

@author: carles
"""

import numpy as np
import imageio
from PIL import Image, ImageFont, ImageDraw

from player import Player
from games import Snake as Snake
from games import Catch as Catch


def play_manually(game):
    while not game.gameover:
        game.draw_screen()
        print("Choose action:", end="")
        a = input()
        r = game.transition(int(a))
        print("Reward:", r)
        print("")

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
        
    BATCH_SIZE = 100
    EPOCHS = 10000
    
    player_params = {"max_epsilon": 0.1,
                     "epochs_to_max_epsilon": 2000,
                     "max_discount":0.95,
                     "epochs_to_max_discount":1000,
                     "kdt":1.0,  # advantage learning k/dt parameter
                     "batch_size":100,
                     "mem_size":500,
                     "win_priority":5,
                     "lose_priority":5,
                     "sur_priority":1,
                     "kernel_initializer":"zeros",
                     "bias_initializer":"ones",
                     "frames_used":4,
                     "convolutional_sizes": ((32, (3, 3)),
                                             (64, (3, 3))),
                     "dense_sizes":(256, 64),
                     "pool_shape":(0, 0),  # not working as of now
                     "dropout":0.1,
                     "learning_rate":0.005,
                     }
                        
    
    
    game = Snake(player_params["frames_used"])
    player = Player(game, **player_params)
                 
#     Uncomment this to play manually
#    play_manually(game)
    
    train(player, game)
    test(player, game)
    
    record_player(player, 'snakegame.gif')
    
    player.save('snakeplayer')
        
