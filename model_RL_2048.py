import json
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import sgd
import jeu_2048


class ExperienceReplay(object):
    # Attributs :
    #   - max memory
    #   - memory --> [[state_t, action_t, reward_t, state_t+1], game_over?]
    #   - discount ?
    def __init__(self, max_memory=100, discount=.9):
        self.max_memory = max_memory
        self.memory = list() # liste des states successifs jusqu'à game over
        self.discount = discount

    # Save les states successifs jusqu'au game over
    def remember(self, states, game_over):
        # memory[i] = [[state_t, action_t, reward_t, state_t+1], game_over?]
        self.memory.append([states, game_over])
        if len(self.memory) > self.max_memory:
            del self.memory[0]

    def get_batch(self, model, batch_size=10):
        len_memory = len(self.memory)
        num_actions = model.output_shape[-1]
        env_dim = self.memory[0][0][0].shape[1]
        inputs = np.zeros((min(len_memory, batch_size), env_dim))
        targets = np.zeros((inputs.shape[0], num_actions))
        for i, idx in enumerate(np.random.randint(0, len_memory,
                                                  size=inputs.shape[0])):
            # Chargement des données des states de la mémoire, successivement
            # et au hasard
            state_t, action_t, reward_t, state_tp1 = self.memory[idx][0]
            game_over = self.memory[idx][1]

            inputs[i:i+1] = state_t
            # There should be no target values for actions not taken.
            # Thou shalt not correct actions not taken #deep
            targets[i] = model.predict(state_t)[0]
            Q_sa = np.max(model.predict(state_tp1)[0])
            if game_over:  # if game_over is True
                targets[i, action_t] = reward_t
            else:
                # reward_t + gamma * max_a' Q(s', a')
                targets[i, action_t] = reward_t + self.discount * Q_sa
        return inputs, targets


def training_bot(epoch=500,
                epsilon=.3,  # exploration
                num_actions=4,  # [move_left, stay, move_right]
                max_memory=500,
                hidden_size=100,
                batch_size=50):
    grid_size=4
    # Modèle DNL
    model = Sequential()
    model.add(Dense(hidden_size, input_shape=(grid_size**2,), activation='relu'))
    model.add(Dense(hidden_size, activation='relu'))
    model.add(Dense(num_actions, activation='softmax'))
    model.compile(sgd(lr=.2), "mse")

    # If you want to continue training from a previous model, just uncomment the line bellow
    model.load_weights("model_2048.h5")

    # Define environment/game
    env = jeu_2048.Game()

    # Initialize experience replay object
    exp_replay = ExperienceReplay(max_memory=max_memory)

    # Train
    win_cnt = 0
    for e in range(epoch):
        loss = 0.
        game_over = False
        # get initial input
        input_t = env.observe()

        while not game_over:
            input_tm1 = input_t
            # get next action
            if np.random.rand() <= epsilon: # Curiosité
                action = np.random.randint(0, num_actions, size=1)
            else:
                q = model.predict(input_tm1)
                action = np.argmax(q[0])

            # apply action, get rewards and new state
            input_t, reward, game_over = env.act(action)

            # store experience
            exp_replay.remember([input_tm1, action, reward, input_t], game_over)

            # adapt model --> apprentissage (memory) ?
            inputs, targets = exp_replay.get_batch(model, batch_size=batch_size)

            loss += model.train_on_batch(inputs, targets)#[0]
        print("Epoch {:03d}/{} | Loss {:.4f} | Win count {}".format(e+1, epoch, loss, win_cnt))

    # Save trained model weights and architecture, this will be used by the visualization code
    model.save_weights("model_2048.h5", overwrite=True)
    with open("model_2048.json", "w") as outfile:
        json.dump(model.to_json(), outfile)


def test_bot(epoch=500,
    with open("model_2048.json", "r") as jfile:
        model = model_from_json(json.load(jfile))
    model.load_weights("model_2048.h5")
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
        cnt_action = 0
        bloque = False
        while not game_over:
            input_tm1 = input_t

            # get next action
            q = model.predict(input_tm1)
            #action = np.argmax(q[0])
            action = np.argmax(q)
            if cnt_action > 4:
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
    plt.savefig('/home/afebvre/RL_bot_Stats_vers01{}_{}_{}_{}.png'.
                format(t.day, t.hour, t.minute, t.second),
                dpi='figure')
    #            transparent=False)
