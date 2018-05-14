import json
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import sgd
import jeu_2048


class Catch(object):
    # Attributs :
    #   - grid size
    #   - state = [[row_fruit, col_fruit, col_mid_panier]]
    def __init__(self, grid_size=10):
        self.grid_size = grid_size
        self.reset()

    # Mise à jour de l'état du jeu (position du fruit et du panier) suite à une
    # action
    def _update_state(self, action):
        """
        Input: action and states
        Ouput: new states and reward
        """
        # Change l'échelle des numéros des actions (de [0 1 2] à [-1 0 1])
        state = self.state
        if action == 0:  # left
            action = -1
        elif action == 1:  # stay
            action = 0
        else:
            action = 1  # right
        f0, f1, basket = state[0]
        new_basket = min(max(1, basket + action), self.grid_size-1)
        f0 += 1
        # Array de sortie après mise à jour (descente du fruit d'1 pixel,
        # déplacement du panier en fonction de l'action
        out = np.asarray([f0, f1, new_basket])
        # ajoute une dimension à l'array en sortie
        out = out[np.newaxis]

        # Vérifie que l'array de sortie est bien de dimension 2
        assert len(out.shape) == 2
        self.state = out

    # Retourne la grille du jeu sous forme d'une matrice (array 2 dim)
    def _draw_state(self):
        im_size = (self.grid_size,)*2
        state = self.state[0]
        canvas = np.zeros(im_size)
        canvas[state[0], state[1]] = 1  # draw fruit
        canvas[-1, state[2]-1:state[2] + 2] = 1  # draw basket
        return canvas

    # - Si le fruit est sur la ligne du panier, à la même abscisse que l'un des
    #   pixels du panier, alors reward = 1, sinon reward = -1
    # - Si le fruit n'est pas sur la ligne du panier : reward = 0
    def _get_reward(self):
        fruit_row, fruit_col, basket = self.state[0]
        if fruit_row == self.grid_size-1:
            if abs(fruit_col - basket) <= 1:
                return 1
            else:
                return -1
        else:
            return 0

    # Regarde dans le state si le game est over ou pas
    def _is_over(self):
        if self.state[0, 0] == self.grid_size-1:
            return True
        else:
            return False

    # Matrice de la grille du jeu sous forme d'un vecteur 1 dim (à partir de
    # _draw_state)
    def observe(self):
        canvas = self._draw_state()
        return canvas.reshape((1, -1))

    # Joue l'action en entrée et retourne la grille (observe --> flat array), la
    # reward, et si le game est over ou pas
    def act(self, action):
        self._update_state(action)
        reward = self._get_reward()
        game_over = self._is_over()
        return self.observe(), reward, game_over

    # Initialise :
    #       - la position du fruit : (row=0, col=n)
    #       - la position du centre du panier (3 pixels de large) : m
    def reset(self):
        n = np.random.randint(0, self.grid_size-1, size=1)
        m = np.random.randint(1, self.grid_size-2, size=1)
        self.state = np.asarray([0, n, m])[np.newaxis]


class ExperienceReplay(object):
    # Attributs :
    #   - max memory
    #   - memory --> [[state_t, action_t, reward_t, state_t+1], game_over?]
    #   - discount ?
    def __init__(self, max_memory=100, discount=.5):
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
        #print(targets)
        return inputs, targets


if __name__ == "__main__":
    # parameters
    epsilon = .1  # exploration
    num_actions = 4  # [gauche, haut, droite, bas]
    epoch = 10
    max_memory = 500
    nb_hidden_layer = 2
    hidden_layer_size = 100
    batch_size = 20
    grid_size = 4

    for i in range(200):
        # Modèle DNL
        model = Sequential()
        model.add(Dense(hidden_layer_size, input_shape=(grid_size**2,), activation='sigmoid'))
        for i in range(nb_hidden_layer):
            model.add(Dense(hidden_layer_size, activation='sigmoid'))
        model.add(Dense(num_actions))
        model.compile(sgd(lr=.2), "mse")

        # If you want to continue training from a previous model, just uncomment the line bellow
        try:
            model.load_weights("model_2048_{}_{}.h5".format(nb_hidden_layer, hidden_layer_size))
            pass
        except OSError as err:
            print("Nouvelle architecture de modèle :")
        print("Nombre de couches total = {} (dont {} cachées)\n"
        "Nombre de neurones par couche = {}".format(
        nb_hidden_layer + 2, nb_hidden_layer, hidden_layer_size))
        # Initialize experience replay object
        exp_replay = ExperienceReplay(max_memory=max_memory)

        # Train
        for e in range(epoch):
            loss = 0.
            max_tile = 0
            game_over = False

            # Define environment/game
            env = jeu_2048.Game()
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

                # print("===========================")
                # print(q)
                # print("action :", ['Gauche', 'Haut', 'Droite', 'Bas'][int(action)], " (", action, ")")
                # apply action, get rewards and new state
                input_t, reward, game_over = env.act(action)
                # print("reward :",reward)
                # print(env.grid)
                if game_over:
                    max_tile = np.max(env.grid)

                # store experience
                exp_replay.remember([input_tm1, action, reward, input_t], game_over)

                # adapt model --> apprentissage (memory) ?
                inputs, targets = exp_replay.get_batch(model, batch_size=batch_size)
                # print(targets)
                loss += model.train_on_batch(inputs, targets)#[0]
            print("Epoch {:03d}/{} | Loss {:.4f} | Max tile {}".format(e+1, epoch, loss, max_tile))

        # Save trained model weights and architecture, this will be used by the visualization code
        model.save_weights("model_2048_{}_{}.h5".format(nb_hidden_layer, hidden_layer_size), overwrite=True)
        with open("model_2048_{}_{}.json".format(nb_hidden_layer, hidden_layer_size), "w") as outfile:
            json.dump(model.to_json(), outfile)
