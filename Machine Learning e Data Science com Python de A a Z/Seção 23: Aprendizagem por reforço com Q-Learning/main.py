from time import sleep
import os, platform
import gym
import random
from IPython.display import clear_output
import numpy as np


def clear():
   if platform.system() == 'Windows':
      os.system('cls')
   else:
      os.system('clear')


env = gym.make('Taxi-v3').env
env.render()

# 0 = south | 1 = move north | 2 = move east | 3 = move west | 4 = pickup passenger | 5 = drop off passenger
print('Quantidade de ações que o taxi pode fazer:', env.action_space)

# 4 destinos
print('Quantidade total de estados:', env.observation_space)
print('Todos os estados possíveis:', env.P)

# Taxa de aprendizagem
alpha = 0.1

# Fator de desconto
gamma = 0.6

# Valor de probabilidade (exploration /exploitation)
epsilon = 0.1

# Armazena o aprendizado do agente
q_table = np.zeros([env.observation_space.n, env.action_space.n])

# TREINAMENTO
for i in range(100000):
    estado = env.reset()
    penalidades, recompensa = 0, 0
    done = False

    while not done:

        # Explorar o ambiente
        if random.uniform(0, 1) < epsilon:
            # Pegar ação randomica
            acao = env.action_space.sample()
        else: # Exploitation
            acao = np.argmax(q_table[estado])

        # Movendo o agente para o próximo estado
        proximo_estado, recompensa, done, info = env.step(acao)

        q_antigo = q_table[estado, acao]

        # Pegando o maior valor do próximo estado
        proximo_maximo = np.max(q_table[proximo_estado])

        q_novo = (1 - alpha) * q_antigo + alpha * (recompensa + gamma * proximo_maximo)
        q_table[estado, acao] = q_novo

        # Levou o passageiro para o lugar errado
        if recompensa == -10:
            penalidades += 1

        estado = proximo_estado

    # Printando a cada 100 episódios
    if i % 100 == 0:
        print('Episódios:', i)

print('Treinamento concluído')


# Resetando o ambiente
env.reset()
env.render()

total_penalidades = 0
episodios = 50

frames = []

for _ in range(episodios):
    estado = env.reset()
    penalidades, recompensa = 0, 0
    done = False

    while not done:
        acao = np.argmax(q_table[estado])
        estado, recompensa, done, info = env.step(acao)

        # Levou o passageiro para o lugar errado
        if recompensa == -10:
            penalidades += 1

        frames.append({
            'frame': env.render(mode='ansi'),
            'state': estado,
            'action': acao,
            'reward': recompensa
        })

        total_penalidades += penalidades

print('Episódios', episodios)
print('Pensalidades', 0)

for frame in frames:
    clear()
    print(frame['frame'])
    print('Estado', frame['state'])
    print('Ação', frame['action'])
    print('Recompensa', frame['reward'])
    sleep(.1)
