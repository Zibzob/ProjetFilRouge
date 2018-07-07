#!/usr/bin/env python3
# coding: utf-8

import os
import time
import pickle
import math
import numpy as np
import seaborn as sns
from datetime import datetime
from collections import deque
import matplotlib.pyplot as plt
from random import random, randint, shuffle, seed, choice
from matplotlib.animation import FuncAnimation

###############################################################################
# Outils
#####################
def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

#def plot_heat(data_matrix, title, fig, plot_agent=False, pos=(4, 4)):
def plot_heat(data):
    data_matrix, title, fig, plot_agent, pos = data
    try:
        ax = fig.gca()
        fig.clf()
    except NameError:
        fig = plt.figure()
        ax = fig.gca()
    sns.heatmap(data_matrix, annot=True)
    plt.title(title)
    # plot l'agent
    if plot_agent:
        agent = plt.Circle(pos, radius=0.5)
        ax.add_artist(agent)
    try:
        sns.plt.show()
    except AttributeError:
        plt.show()

###############################################################################
# Environment
#####################
class environnement:
    def __init__(self, n1, n2, end=None, pos=None):
        self.grid = np.zeros((n1, n2), dtype='int')
        self.shape = (n1, n2)
        # Case victoire
        self.end = (randint(0, n1-1), randint(0, n2-1)) if end == None else end
        # Position initiale
        self.pos = (randint(0, n1-1), randint(0, n2-1)) if pos == None else pos
        while self.pos == self.end:
            self.pos = (randint(0, n1-1), randint(0, n2-1))
        self.game_over = False

    def re_init(self):
        n1, n2 = self.shape
        self.pos = (randint(0, n1-1), randint(0, n2-1)) # Position initiale
        while self.pos == self.end:
            self.pos = (randint(0, n1-1), randint(0, n2-1))
        self.game_over = False

    def move(self, a, bool_plot=False):
        moves = [(0, -1), (-1, 0), (0, 1), (1, 0)]
        n1, n2 = self.shape
        x, y = self.pos
        # Mur horizontal sur la 3eme ligne de la première à l'avant-avant der col
        mur = True
        if mur and x == 3 and y < n2 - 2:
            self.pos = (np.min([n1-1, np.max([3, x + moves[a][0]])]),
                        np.min([n2-1, np.max([0, y + moves[a][1]])]))
        elif mur and x == 2 and y < n2 - 2:
            self.pos = (np.min([2, np.max([0, x + moves[a][0]])]),
                        np.min([n2-1, np.max([0, y + moves[a][1]])]))
        # Grille normale
        else:
            self.pos = (np.min([n1-1, np.max([0, x + moves[a][0]])]),
                        np.min([n2-1, np.max([0, y + moves[a][1]])]))
            vent = 0
            if vent:
                # Vent (décalage d'une case dans le sens du vent quand l'agent arrive
                # sur une case vent
                vent = True
                x, y = self.pos
                if vent and (x, y) == (5, 2):
                    self.pos = (x-1, y)
                elif vent and (x, y) in [(4, 1), (5, 1), (6, 1), (6, 2), (6, 3), (6, 4), (5, 3), (5, 4)]:
                    self.pos = (x+1, y)
                elif vent and (x, y) in [(4, 2), (4, 3), (4, 4)]:
                    self.pos = (x, y+1)

        if self.pos == self.end:
            self.game_over = True
            reward = -1
        else:
            reward = -1
        if bool_plot:
            self.plot()
            #time.sleep(0.2)

        return reward

    def plot(self):
        instantane = self.grid.copy()
        x, y = self.pos
        xe, ye = self.end
        instantane[x, y] = 1
        instantane[xe, ye] = 2
        os.system('clear') # unix
        #os.system('cls') # windows
        print(instantane)
        
    def plot_game(self, mem):
        """Affiche simplement le parcours suivi pendant une partie en ligne de
        commande"""
        self.grid[self.end] = 2
        for s, a, r, s1 in mem:
            self.pos = s
            self.grid[s] = 1
            time.sleep(0.1)
            self.plot()
            self.grid[s] = 0
        print(len(mem))

    def play_bot(self, q, pos_init):
        self.pos = pos_init
        while self.pos != self.end:
            x, y = self.pos
            a = np.argmax([q[x, y, ai] for ai in range(num_actions)])
            self.move(a, True)

###############################################################################
# RL algorithms ("Reinforcement Learning - An Introduction, second edition Sutton and Barto")
##################### Dynamic Programming
def value_iteration(version, env, epoch, gama, plot_process=False, gif=False):
    """Achieve v* by iterating over Bellman optimality backup"""
    n1, n2 = env.shape
    v = np.zeros((n1, n2))
    # 1) update every state at each iteration with current state values
    if version == 1:
        fig1 = plt.figure("Value iteration - Maj t")
        fig1_gif = plt.figure("Value iteration - Maj t")
        vt, vt1 = v.copy(), v.copy()
        l_gif = [(vt, "Itération n°{} sur {}".format(0, epoch), fig1_gif, False, None)] # enregistre les valeurs des arguments de plot_heat pour chaque itération
        for i in range(epoch):
            for j in range(n1):
                for k in range(n2):
                    if (j, k) == env.end:
                        vt1[j, k] = 0
                    else:
                        rs = []
                        for ai in range(4):
                            # modelisation de notre environnement
                            model = environnement(n1, n2, end=env.end, pos=(j, k))
                            rs.append((model.move(ai), model.pos))
                        vt1[j, k] = np.max([r + gama*vt[x, y] for r, (x, y) in rs])
            vt = vt1.copy()
            l_gif.append((vt, "Itération n°{} sur {}".format(i+1, epoch), fig1_gif, False, None))
            if plot_process:
                plot_heat((vt, "Itération n°{} sur {}".format(i, epoch), fig1, False, None))
                plt.draw()
                plt.pause(0.5)

        # Save un gif sur l'ensemble des epoch
        if gif:
            anim = FuncAnimation(fig1_gif, plot_heat, frames=l_gif, interval=1000)
            anim.save('value_iteration_t_complete.gif', dpi=80, writer='imagemagick')
        # HeatMap final
        title_hm = "Value iteration : complete at each iteration"
        plot_heat((vt, title_hm, fig1, False, None))
        fig1.savefig('VI final state 1')

    # 2) update every state at each iteration (with t or t+1 state values)
    if version == 2:
        fig2 = plt.figure("Value iteration - Maj t ou t+1")
        fig2_gif = plt.figure("Value iteration - Maj t ou t+1")
        vt = v.copy()
        l_gif = [(vt.copy(), "Itération n°{} sur {}".format(0, epoch), fig2_gif, False, None)] # enregistre les valeurs des arguments de plot_heat pour chaque itération
        for i in range(epoch):
            for j in range(n1):
                for k in range(n2):
                    if (j, k) == env.end:
                        vt[j, k] = 0
                    else:
                        rs = []
                        for ai in range(4):
                            # modelisation de notre environnement
                            model = environnement(n1, n2, end=env.end, pos=(j, k))
                            rs.append((model.move(ai), model.pos))
                        vt[j, k] = np.max([r + gama*vt[x, y] for r, (x, y) in rs])
            l_gif.append((vt.copy(), "Itération n°{} sur {}".format(i+1, epoch), fig2_gif, False, None))
            if plot_process:
                plot_heat((vt, "Itération n°{} sur {}".format(i, epoch), fig2, False, None))
                plt.draw()
                plt.pause(0.5)
        # Save un gif sur l'ensemble des epoch
        if gif:
            anim = FuncAnimation(fig2_gif, plot_heat, frames=l_gif, interval=1000)
            anim.save('value_iteration_t_tp1.gif', dpi=80, writer='imagemagick')
        # HeatMap final
        title_hm = "Value iteration : overwrite"
        plot_heat((vt, title_hm, fig2, False, None))
        fig2.savefig('VI final state 2')

##################### Monte Carlo
def MC_pi_eval(pi, env, epoch, epsilon=0.3, gama=1, max_reward=10, bool_plot=False):
    """Evalue q value, Monte Carlo method, exploring start"""
    n1, n2 = env.shape
    num_actions = 4
    # List of all returns for all states
    S = [(i, j) for i in range(n1) for j in range(n2)]
    G_s_a = {(s, a):[] for s in S for a in range(num_actions)}
    # Init q(s, a), exploring start
    q = np.zeros(shape=(n1, n2, num_actions)) + max_reward
    xf, yf = env.end
    q[xf, yf, :] = 0

    stats = []
    for i in range(epoch):
        eps = epsilon / (i+1)**0.5
        # Initialisation random du couple state/action de départ
        env.re_init()
        curio = True
        # Mémoire de lambda states successifs
        mem = []
        while not env.game_over:
            s = env.pos
            x, y = s
            if curio:
                # Curiosity
                a = randint(0, num_actions-1)
            else:
                # Policy pi
                a = pi(s, q, eps)
            r = env.move(a, bool_plot=bool_plot)
            s1 = env.pos
            mem.append((s, a, r, s1))
            curio = False

        # Mise à jour modèle
        for (s, a, r, s1) in mem[-1::-1]:
            # Episode's return
            G = r if s1 == env.end else gama*G + r
            # All returns for state s and action a
            G_s_a[s, a].append(G)
            x, y = s
            q[x, y, a] = np.mean(G_s_a[s, a])

        stats.append(len(mem))
    plt.plot(range(epoch), stats)
    plt.show()
    print(np.mean(stats[500:]))

    V = np.matrix([[round(np.max([q[i, j, k] for k in range(num_actions)]), 2) for j in range(n2)] for i in range(n1)])
    title_hm = "MC policy ({}) evaluation - exploring start".format(pi)
    fig = plt.figure()
    plot_heat((V, title_hm, fig, False, None))

    return q, G_s_a

def MC_ES(pi, env, epoch, bool_plot=False):
    """Monte Carlo Exploring Start (on policy-method), for estimating pi ~ pi*"""
    n1, n2 = env.shape
    num_actions = 4
    S = [(i, j) for i in range(n1) for j in range(n2)]
    # 1) Exploring start
    end = env.end
    # List of all returns for all states
    gt_s = {s:[] for s in S}
    # init q(s, a)
    q = np.zeros(shape=(n1, n2, num_actions))
    for i in range(epoch):
        # Initialisation random du couple state/action de départ
        env.re_init()
        curio = True
        # Mémoire de lambda states successifs
        mem = []
        while not env.game_over:
            s = env.pos
            x, y = s
            if curio:
                # Curiosité
                a = randint(0, num_actions-1)
            else:
                # pi
                a = pi(s, q)
            r = env.move(a, bool_plot=bool_plot)
            s1 = env.pos
            mem.append((s, a, r, s1))
            curio = False

        # Mise à jour modèle
        for (s, a, r, s1) in mem[-1::-1]:
            if s1 == env.end:
                gt_s[s1] = [0]
            gt_s[s].append(r + gt_s[s1][-1])
            x, y = s
            q[x, y, a] = np.mean(gt_s[s])

    V = np.matrix([[round(np.max([q[i, j, k] for k in range(num_actions)]), 2) for j in range(n2)] for i in range(n1)])
    title_hm = "MC policy evaluation - exploring start"
    fig = plt.figure()
    plot_heat((V, title_hm, fig, False, None))

    return q

##################### Temporal Difference (TD)
# n step TD - SARSA
def n_step_SARSA(n, epoch, q_file_name, gama=1.0, alpha=0.2, epsilon=0.1, bool_plot=False):
    """N step SARSA prevision/control on policy method. """

    # Total return of the n-long sequence
    def Gt_n(mem):
        if len(mem) == 1:
            s, a, r, (x1, y1), a1 = mem.popleft()
            res = r + gama*q[x1, y1, a1]
            return res
        else:
            s, a, r, s1, a1 = mem.popleft()
            res = r + gama*Gt_n(mem)
            return res

    # Initialisation de q(s, a)
    n1, n2 = env.shape
    num_actions = 4
    try:
        q = load_obj(q_file_name)
        print("Poursuite training modèle existant")
    except OSError as err:
        print("Nouveau modèle")
        q = np.zeros(shape=(n1, n2, num_actions))
        x, y = env.end
    # Play several games in a row (from random starting positions) and update q
    for i in range(epoch):
        # Curiosity decay
        eps = epsilon / (i+1)**0.5
        # Future doubt decay (gama --> 1)
        #gama = 1 - (1 - gama)*1/np.log(i+math.e)
        # n-state memory (saves [s(t), a(t), r(t+1), s(t+1)] at each time step)
        mem = deque(maxlen=n)
        # Environement initialisation
        env.re_init()
        print(env.pos)
        # Game progress...
        while not env.game_over:
            # Current state : t
            s = x, y = env.pos
            curio = random() < eps
            # Curiosity
            if curio:
                a = randint(0, num_actions-1)
            # Greed-on-q policy ("on-policy")
            else:
                a = np.argmax([q[x, y, action] for action in range(num_actions)])
            r1 = env.move(a, bool_plot=bool_plot)
            # Next state : t+1
            s1 = x1, y1 = env.pos
            a1 = np.argmax([q[x1, y1, action] for action in range(num_actions)])
            mem.append((s, a, r1, s1, a1))

            # Update q-value of state t-n (and every state between T-n and T, T
            # being the final state)
            if len(mem) == n or env.game_over:
                mem_bis = mem.copy()
                s_maj, a_maj, r1_maj, s1_maj, a1_maj = mem_bis[0]
                x_maj, y_maj = s_maj
                # Total return (for the n-long sequence of states/actions)
                gt = Gt_n(mem_bis)
                q[x_maj, y_maj, a_maj] = q[x_maj, y_maj, a_maj] + alpha*(gt - q[x_maj, y_maj, a_maj])
                # If curiosity, update only if policy is improved
                #if curio or (gt - q[x_maj, y_maj, a_maj]) < 0:
                #    q[x_maj, y_maj, a_maj] = q[x_maj, y_maj, a_maj] + alpha*(gt - q[x_maj, y_maj, a_maj])

    return q

##################### In between : SARSA, Q-learning, TD(lambda)
def TD_lambda(env, model_file_name=None, epoch=10, epsilon=0.1, alpha=0.3, lam=5, gama=0.8,
              num_actions=4, bool_plot=False):
    """Evalue q value, TDlambda method"""

    # TD_lambda : total return of the lambda-long sequence
    def Gt_lam(mem_bis):
        if len(mem_bis) == 1:
            s, a, r, s1 = mem_bis.popleft()
            x1, y1 = s1
            a1 = np.argmax([q[x1, y1, ai] for ai in range(num_actions)])
            res = r + gama*q[x1, y1, a1]
            mem_bis.appendleft((s, a, r, s1))
            return res
        else:
            s, a, r, s1 = mem_bis.popleft()
            res = r + gama*Gt_lam(mem_bis)
            mem_bis.appendleft((s, a, r, s1))
            return res

    # Initialisation de q(s, a)
    n1, n2 = env.shape
    try:
        q = load_obj(model_file_name)
        print("Poursuite training modèle existant")
    except OSError as err:
        print("Nouveau modèle")
        q = np.zeros(shape=(n1, n2, num_actions))
        x, y = env.end
    # Joue plusieurs parties successives (pos initiale aléatoire) en mettant à 
    # jour q
    for i in range(epoch):
        # Curiosity decay
        #eps = epsilon / (i+1)**0.5
        # Future doubt decay (gama --> 1)
        gama = 1 - (1 - gama)*1/np.log(i+math.e)
        # Initialisation partie et state
        env.re_init()
        s = env.pos
        # Mémoire de lambda states successifs
        mem = deque(maxlen=lam)
        while not env.game_over:
            x, y = s
            curio = random() < eps
            if curio:
                # Curiosité
                a = randint(0, num_actions-1)
            else:
                # Greed
                a = np.argmax([q[x, y, action] for action in range(num_actions)])
            r = env.move(a, bool_plot=bool_plot)
            s1 = env.pos
            mem.append((s, a, r, s1))

            # Mise à jour modèle (des lambda states précédents) : learning, dès
            # qu'on a une séquence complète (longueur lambda) ou fini la partie
            mem_bis = mem.copy()
            if len(mem_bis) == lam or env.game_over:
                while len(mem_bis) != 0:
                    gt = Gt_lam(mem_bis)
                    s_maj, a_maj, r_maj, s1_maj = mem_bis[-1]
                    x_maj, y_maj = s_maj
                    # Quand curiosité, maj du modèle uniquement si ça améliore
                    if not(curio and (gt - q[x_maj, y_maj, a_maj]) < 0):
                        q[x_maj, y_maj, a_maj] = q[x_maj, y_maj, a_maj] + alpha*(gt - q[x_maj, y_maj, a_maj])
                    mem_bis.popleft()

            s = env.pos

    return q

###############################################################################
# Policies
#####################
def pi_rand(s, q, eps):
    """Random policy (action associated to s following pi policy)"""
    return randint(0, 3)
    
def pi_eps_greedy(s, q, eps):
    """Greedy policy with explo"""
    x, y = s
    curio = random() < eps
    if curio:
        a = randint(0, 3)
    else:
        # Greed, but random between ties
        q_s = [q[x, y, a] for a in range(4)]
        ind_maxs_q = []
        ind_q_sorted = np.argsort(q_s)[-1::-1]
        q0 = q_s[ind_q_sorted[0]]
        for i in ind_q_sorted:
            if q_s[i] != q0:
                break
            ind_maxs_q.append(i)
            q0 = q_s[i]
        a = choice(ind_maxs_q)

    return a

###############################################################################
# MAIN
#####################
if __name__ == '__main__':
    seed(1)

    ###################
    # Value iteration & MC policy evaluation
    if 1:
        # Paramètres
        n1, n2 = 8, 6
        n = 3
        epoch = 2
        epsilon = 0.1
        alpha = 0.2
        gama = 1.0
        num_actions = 4

        # Environement creation
        env = environnement(n1, n2, end=(1, 2))

        # Algos
        value_iteration(1, env, 15, gama=1, plot_process=False, gif=False)
        #q = MC_pi_eval(pi_rand, env, epoch, bool_plot=False)
        #q, G = MC_pi_eval(pi_eps_greedy, env, epoch, bool_plot=False)
        #q_dir = np.matrix([[round(np.argmax([q[i, j, k] for k in range(4)]), 2) for j in range(n2)] for i in range(n1)])
        q_file_name = "sarsa_n_step1"
        q = n_step_SARSA(n, epoch, q_file_name, gama, alpha, epsilon)
        #save_obj(q, q_file_name)

        if 1:
            # Plot V at the end of the training
            n1, n2, l = q.shape
            V = np.matrix([[round(np.max([q[i, j, k] for k in range(l)]), 2) for j in range(n2)] for i in range(n1)])
            title_hm = ("n1={}, n2={}, lambda={}, epoch={}\neps={}, alpha={}, gama={},"
                              "nb_a={}".format(n1, n2, n, epoch, epsilon, alpha, gama, num_actions))
            fig = plt.figure()
            plot_heat((V, title_hm, fig, False, None))
    ###################
    ###################
    ###################
    ################### voir quel critère/visu utiliser pour pouvoir comparer les différents algos
    ###################
    ###################
    ###################
    ###################


    ###################
    # TD lambda (pour l'instant)
    if 0:
        for lam in [1, 4]:
        #for lam in [1, 2, 3, 4, 5, 7, 10, 20, 40, 70, 100, 1000]:
        #for epoch in [10, 100, 1000, 10000]:
            # Paramètres
            n1, n2 = 5, 8
            #lam = 1
            epoch = 100
            epsilon = 0.5
            alpha = 0.3
            gama = 0.9
            num_actions = 4  # [gauche, haut, droite, bas]

            # Environement creation
            env = environnement(n1, n2, end=(1, 2))
            t0 = time.time()
            # Load the model if already exists
            model_file_name = "modele_1"
            # Learn model
            q = TD_lambda(env, model_file_name, epoch, epsilon, alpha, lam, gama, num_actions, False)
            # Save model
            save_obj(q, model_file_name)
            print("Temps tot pour lambda = {} : {} secondes".format(lam, time.time()-t0))

            # Plot de V après n epochs pour les states de la grille
            n1, n2, l = q.shape
            V = np.matrix([[round(np.max([q[i, j, k] for k in range(l)]), 2) for j in range(n2)] for i in range(n1)])
            title_hm = ("n1={}, n2={}, lambda={}, epoch={}\neps={}, alpha={}, gama={},"
                              "nb_a={}".format(n1, n2, lam, epoch, epsilon, alpha, gama, num_actions))
            plot_heat(V, title_hm)
            t = datetime.now()
            file_name = ('ImageRes/Toy/HeatMap_{}_{}_{}_{}_{}.png'.
                         format(t.day, t.hour, t.minute, t.second, t.microsecond))
            hm.savefig(file_name, dpi='figure')
            plt.clf()
        # Display step by step the path given q_pi function and starting position
        #env.play_bot(q, (14, 0))


