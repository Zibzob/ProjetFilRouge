#!/usr/bin/env python3
# coding: utf-8

from random import random, randint, shuffle
import numpy as np

def push_left(grid):
    moved, score = False, 0
    rows, columns = grid.shape[0], grid.shape[1]
    for k in range(rows):
        i, last = 0, 0
        for j in range(columns):
            e = grid[k, j]
            if e:
                if e == last:
                    grid[k, i-1]+=e
                    score += e
                    last, moved = 0, True
                else:
                    moved |= (i != j)
                    last=grid[k, i]=e
                    i+=1
        while i<columns:
            grid[k,i]=0
            i+=1
    return score if moved else -1

def push_right(grid):
    moved, score = False, 0
    rows, columns = grid.shape[0], grid.shape[1]
    for k in range(rows):
        i = columns-1
        last  = 0
        for j in range(columns-1,-1,-1):
            e = grid[k, j]
            if e:
                if e == last:
                    grid[k, i+1]+=e
                    score += e
                    last, moved = 0, True
                else:
                    moved |= (i != j)
                    last=grid[k, i]=e
                    i-=1
        while 0<=i:
            grid[k, i]=0
            i-=1
    return score if moved else -1

def push_up(grid):
    moved,score = False, 0
    rows, columns = grid.shape[0], grid.shape[1]
    for k in range(columns):
        i, last = 0, 0
        for j in range(rows):
            e = grid[j, k]
            if e:
                if e == last:
                    score += e
                    grid[i-1, k]+=e
                    last, moved = 0, True
                else:
                    moved |= (i != j)
                    last=grid[i, k]=e
                    i+=1
        while i<rows:
            grid[i, k]=0
            i+=1
    return score if moved else -1

def push_down(grid):
    moved, score = False, 0
    rows, columns = grid.shape[0], grid.shape[1]
    for k in range(columns):
        i, last = rows-1, 0
        for j in range(rows-1,-1,-1):
            e = grid[j, k]
            if e:
                if e == last:
                    score += e
                    grid[i+1, k]+=e
                    last, moved = 0, True
                else:
                    moved |= (i != j)
                    last=grid[i, k]=e
                    i-=1
        while 0<=i:
            grid[i, k]=0
            i-=1
    return score if moved else -1

# & compare bitwise, du coup : 
#       - 0 = gauche
#       - 1 = haut
#       - 2 = droite
#       - 3 = bas
def push(grid, direction):
    if direction&1:
        if direction&2:
            score = push_down(grid)
        else:
            score = push_up(grid)
    else:
        if direction&2:
            score = push_right(grid)
        else:
            score = push_left(grid)
    return score

def put_new_cell(grid):
    n = 0
    r = 0
    i_s=[0]*16
    j_s=[0]*16
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            if not grid[i,j]:
                i_s[n]=i
                j_s[n]=j
                n+=1
    if n > 0:
        r = randint(0, n-1)
        grid[i_s[r], j_s[r]] = 2 if random() < 0.9 else 4
    return n

def any_possible_moves(grid):
    """Return True if there are any legal moves, and False otherwise."""
    rows = grid.shape[0]
    columns = grid.shape[1]
    for i in range(rows):
        for j in range(columns):
            e = grid[i, j]
            if not e:
                return True
            if j and e == grid[i, j-1]:
                return True
            if i and e == grid[i-1, j]:
                return True
    return False

def prepare_next_turn(grid):
    """
    Spawn a new number on the grid; then return the result of
    any_possible_moves after this change has been made.
    """
    empties = put_new_cell(grid)    
    return empties>1 or any_possible_moves(grid)

def print_grid(grid_array):
    """Print a pretty grid to the screen."""
    print("")
    wall = "+------"*grid_array.shape[1]+"+"
    print(wall)
    for i in range(grid_array.shape[0]):
        meat = "|".join("{:^6}".format(grid_array[i,j]) for j in range(grid_array.shape[1]))
        print("|{}|".format(meat))
        print(wall)

def plot_grid(grid_array):
    pass

class Game:
    def __init__(self, cols=4, rows=4):
        #self.grid_array = np.zeros(shape=(rows, cols), dtype='uint16')
        self.grid_array = np.zeros(shape=(rows, cols), dtype='float')
        self.grid = self.grid_array
        for i in range(2):
            put_new_cell(self.grid)
        self.score = 0
        self.diff_score = 0 # score de l'action, a ajouter au score global
        self.memory_action = -1
        self.same_move = False
        self.end = False
    
    def copy(self):
        rtn = Game(self.grid.shape[0], self.grid.shape[1])
        for i in range(self.grid.shape[0]):
            for j in range(self.grid.shape[1]):
                rtn.grid[i,j]=self.grid[i,j]
        rtn.score = self.score
        rtn.end = self.end
        return rtn

    def max(self):
        m = 0
        for i in range(self.grid.shape[0]):
            for j in range(self.grid.shape[1]):
                if self.grid[i,j]>m:
                    m = self.grid[i,j]
        return m
        

    def move(self, direction):
        if direction&1:
            if direction&2:
                score = push_down(self.grid)
            else:
                score = push_up(self.grid)
        else:
            if direction&2:
                score = push_right(self.grid)
            else:
                score = push_left(self.grid)
        if direction == self.memory_action:
            self.same_move = True
        else:
            self.same_move = False
        self.memory_action = direction
        # gestion score et game over
        self.score += score
        self.diff_score = score
        # print(score)
        if score == -1:
            return 0
        if not prepare_next_turn(self.grid):
            self.end = True
        return 1

    def display(self):
        print_grid(self.grid_array)

    def observe(self):
        """Returns the current state of the grid in a flat array"""
        canvas = self.grid
        return canvas.reshape((1, -1))

    def _get_reward(self):
        if any_possible_moves(self.grid):
            return self.diff_score if not self.same_move else self.diff_score - 1
        else:
            return -1000

    def act(self, action):
        self.move(action)
        reward = self._get_reward()
        game_over = not(any_possible_moves(self.grid))
        return self.observe(), reward, game_over

def random_play(game):
    moves = [0,1,2,3]
    moves_count = 0
    while not game.end:
        moves_count += 1
        shuffle(moves)
        for m in moves:
            if game.move(m):
                    break    
    return game.score, moves_count

# MAIN
# =============================================================================
if __name__ == '__main__':
    game = Game()
    random_play(game)
