#!/usr/bin/env python3
# coding: utf-8

import time
from datetime import datetime
import numpy as np
import seaborn as sns
import jeu_2048 as jeu
import matplotlib.pyplot as plt

# FUNCTIONS - CLASSES
# =============================================================================
# STATISTIQUES
def stat_random_games(n):
    """Plays n games of 2048 with random moves and returns statistics about
    it"""
    score = []
    count = []
    max_tile = []
    memoire = 0
    for i in range(n):
        avancement = i*100//n
        if avancement != memoire:
            memoire = avancement
            print(avancement, "% effectués")
            print(len(score), len(count), len(max_tile))
        game = jeu.Game()
        s, c = jeu.random_play(game)
        score.append(s)
        count.append(c)
        max_tile.append(np.max(game.grid))

    tiles, nb = np.unique(max_tile, return_counts=True)
    # Plots and stats
    f = plt.figure("Res random", figsize=(8, 6), dpi=400)
    # f = plt.figure("Res random", dpi=100)
    # Scores
    plt.subplot(221)
    sns.distplot(score, kde=True, rug=True);
    plt.xlabel("Score")
    plt.ylabel("Density")
    plt.title("Score distribution")
    # Maxtiles
    plt.subplot(222)
    sns.countplot(max_tile);
    plt.xlabel("Max tile")
    plt.xticks(rotation=55)
    plt.ylabel("Count")
    plt.title("Max tile distribution")
    # Moves count
    plt.subplot(223)
    sns.distplot(count, rug=True);
    plt.xlabel("Moves count")
    plt.ylabel("Density")
    plt.title("Moves count distribution")

    # Print and save
    plt.tight_layout()
    t = datetime.now()
    #plt.show()
    #plt.savefig('/home/aurelien/Pictures/pfr/RandGameStats_{}_{}_{}_{}.png'.
    plt.savefig('/home/afebvre/RandGameStats_{}_{}_{}_{}.png'.
                format(t.day, t.hour, t.minute, t.second),
                dpi='figure')
    #            transparent=False)
    #plt.close('all')


# MAIN
# =============================================================================
if __name__ == '__main__':
    t1 = time.time()
    stat_random_games(100000)
    print("Temps écoulé : ", (time.time() - t1) / 60, ' minutes')
