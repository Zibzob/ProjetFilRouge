from datetime import datetime
from random import randint
import json
import matplotlib.pyplot as plt
import numpy as np
from keras.models import model_from_json
import jeu_2048
import seaborn as sns


if __name__ == "__main__":
    # Make sure this grid size matches the value used fro training

    nb_hidden_layer = 2
    hidden_layer_size = 100
    model_name = "model_2048_{}_{}".format(nb_hidden_layer, hidden_layer_size)
    with open(model_name + ".json", "r") as jfile:
        model = model_from_json(json.load(jfile))
    model.load_weights(model_name + ".h5")
    model.compile("sgd", "mse")

    # Define environment, game
    score = []
    max_tile = []
    count = []
    for e in range(100):
        env = jeu_2048.Game()
        loss = 0.
        game_over = False
        # get initial input
        input_t = env.observe()

        cnt = 0
        mem_action = -1
        cnt_action = 1
        bloque = False
        stat_blocked = 0
        while not game_over:
            input_tm1 = input_t

            # get next action
            q = model.predict(input_tm1)
            #action = np.argmax(q[0])
            action = np.argmax(q)
            if cnt_action >= 2:
                stat_blocked += 1
                action = randint(0, 3)
                cnt_action = 0
            if action == mem_action:
                cnt_action += 1
            mem_action = action

            # apply action, get rewards and new state
            input_t, reward, game_over = env.act(action)

            # Critères de réussite
            s = env.score
            max_t = np.max(env.grid)
            cnt += 1

        score.append(s)
        max_tile.append(max_t)
        count.append(cnt)
        print("Nombre de random pour debloquer = {} (sur {} actions --> {}%)".
        format(stat_blocked, cnt, round(stat_blocked/cnt * 100, 1)))

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
    #plt.savefig('/home/aurelien/Pictures/pfr/RandGameStats_{}_{}_{}_{}.png'
    plt.savefig('/home/aurelien/Documents/CoursINSA/PFR/Image_res_bots/RL_bot_Stats_vers_{}_{}_{}_{}_{}_{}.png'.
                format(nb_hidden_layer, hidden_layer_size, t.day, t.hour,
                t.minute, t.second),
                dpi='figure')
    #            transparent=False)
    #plt.close('all')
