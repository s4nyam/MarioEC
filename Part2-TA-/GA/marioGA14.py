
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import *
import matplotlib.pyplot as plt
from matplotlib import animation, rc
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
import random, sys
import numpy as np
import os
import time



# Create a folder with a timestamp as its name
timestamp = int(time.time())
output_folder = f"generations_{timestamp}"
os.makedirs(output_folder)

plt.ioff()
env = gym_super_mario_bros.make('SuperMarioBros-1-4-v0')
env = JoypadSpace(env, RIGHT_ONLY)

# Animation
save_animation = 1  # Saves animation to plot
frames_skipped = 1  # How many frames to skip when eating animation

# Generation
generation_amount = 1000

# Population
population_amount = 10 # Amount of agents per generation

# Agent
moves_amount = 5000  # Max amount of moves an agent can perform
moves_to_check = 30  # How many moves without an increase in fitness until termination
mutation_rate = 0.8  # Chance of any move changing to a random move
moves_mutable = 0.8  # How many percent of moves are mutable (starting from the end)


coin_reward = 30.0
distance_reward = 0.1
time_penalty = 0.01
# Add constraints for Max time which Mario can stay alive, rebirth constraint
# Add max number of births that an algorithm can take: Rebirth Constraint
# Add max amount of time that an algorithm can utilize: Time Constraint
# Move Constraint already implemented

t = 0
fitness_history = []  # To store the best fitness from each generation

def print_info(info, fitness):
    global t
    t += 1
    if not t % 100:
        print(info, fitness)

def handle_frame(step, images):
    if not step % frames_skipped:
        image = plt.imshow(env.render(mode='rgb_array'))
        images.append([image])

fig = plt.figure()

def display_animation(images):
    anim = animation.ArtistAnimation(fig, images,
                                     interval=15 * frames_skipped, blit=True)
    rc('animation', html='jshtml')
    return anim

class Agent:
    def __init__(self, fittest):
        self.fitness = 0
        self.images = []
        if fittest:
            self.moves = fittest.moves.copy()
            self.moves_used = fittest.moves_used
        else:
            self.moves = []
            self.moves_used = 0

    def randomize_moves(self):
        self.moves = [env.action_space.sample() for _ in range(moves_amount)]

def create_population():
    population = []
    for _ in range(population_amount):
        agent = Agent(None)
        agent.randomize_moves()
        population.append(agent)
    return population

# Define a new fitness that aligns the part 1 of the report.
# Maximize the total coin value collected while minimizing the total distance traveled:
# Maximize: n∑ i=1 coini − λ n∑ i=1 distancei Where: • coini
# represents the value of the i-th collected coin. • distancei represents the distance associated with
# the i-th action. • λ is a trade-off parameter that determines the balance between maximizing coins and minimizing distance.
# Adjusting this parameter allows controlling the importance of each objective
import math

def check_fitness(player, info, past_fitness):
    coins_collected = info['coins']
    distance = info['x_pos']
    time_left = info['time']
    
    # You can adjust the trade-off parameters to balance coins, distance, and time

    
    # Calculate fitness as a combination of coins collected, distance traveled, and time survived
    fitness = (coin_reward * coins_collected + distance_reward * distance) - time_penalty * (300 - time_left)
    # In this formula:
    # coin_reward is a positive reward for each coin collected.
    # distance_reward encourages covering more distance.
    # time_penalty penalizes the agent for taking too much time.
    
    if player.fitness < fitness:
        player.fitness = fitness
    
    # Add the fitness to past_fitness
    past_fitness.append(fitness)
    
    # Check if there has been no progress in fitness for moves_to_check moves
    if len(past_fitness) > moves_to_check:
        past_fitness.pop(0)
        for i in range(moves_to_check):
            if past_fitness[i] > past_fitness[0]:
                break
        # If no progress, terminate the agent
        if i == moves_to_check - 1:
            return True
    
    return False




def mutate_moves(player):
    start_index = int(player.moves_used * (1 - moves_mutable))
    if player.moves_used - start_index < moves_to_check * 2:
        start_index = player.moves_used - moves_to_check * 2
    for i in range(start_index, moves_amount):
        num = random.random()
        if num < mutation_rate:
            player.moves[i] = env.action_space.sample()

def agent_get_moves(player, population):
    if not player.moves:
        for i in range(moves_amount):
            player.moves.append(env.action_space.sample())
    else:
        if population:
            mutate_moves(player)

def print_mutation_info(fittest, population):
    global mutation_rate
    current_fittest = get_fittest(population)
    mutable_amount = current_fittest.moves_used * moves_mutable  # new line
    if fittest and fittest.fitness == current_fittest.fitness:
        print("Increasing mutation chance by 0.005")
        mutation_rate += 0.005
        print("Mutation chance is now {}%".format(mutation_rate * 100))
    else:
        print("Fitness increased successfully, restore mutation chance to 0.01")
        mutation_rate = 0.01
        mutable_amount = current_fittest.moves_used * moves_mutable

    if mutable_amount < moves_to_check * 2:
        mutable_amount = moves_to_check * 2

    print("Mutating the last {}% of moves, equal to {} moves".format(moves_mutable * 100,
                                                                    mutable_amount))

def display_recording(player):
    print("Displaying player with fitness score {}.".format(
        player.fitness))
    display_animation(player)

def get_fittest(population):
    player_num = 0
    for i in range(1, len(population)):
        if population[i].fitness > population[player_num].fitness:
            player_num = i
    return population[player_num]

def record_player(player):
    images = []
    done = False
    observation = env.reset()
    past_fitness = []
    for move in range(len(player.moves)):
        if done:
            break
        state, fitness, done, info = env.step(player.moves[move])
        # state, fitness, done,info = env.step(player.moves[move])
        if done:
            break
        done = check_fitness(player, info, past_fitness)
        handle_frame(move, images)
    return images

# Tournament selection used
def play_generation(fittest, population):
    # Create a copy of the fittest agent without any changes
    elite_agent = Agent(fittest)
    observation = env.reset()
    done = False
    past_fitness = []
    agent_get_moves(elite_agent, population)
    for move in range(len(elite_agent.moves)):
        if done:
            break
        state, fitness, done, info = env.step(elite_agent.moves[move])
        if done:
            break
        done = check_fitness(elite_agent, info, past_fitness)
        env.render()
    elite_agent.moves_used = move

    # Preserve the elite agent without any changes
    elite_fitness = elite_agent.fitness

    # Apply crossover and mutation to the rest of the population
    for i in range(1, population_amount):
        player = Agent(fittest)
        observation = env.reset()
        done = False
        past_fitness = []
        agent_get_moves(player, population)
        for move in range(len(player.moves)):
            if done:
                break
            state, fitness, done, info = env.step(player.moves[move])
            if done:
                break
            done = check_fitness(player, info, past_fitness)
            env.render()
        player.moves_used = move
        population.append(player)
        print("Player {} achieved a fitness of {} in {} moves".format(i,
                                                                      player.fitness, move))
        if info['flag_get']:
            return 1

    print_mutation_info(fittest, population)

    # Restore the elite agent in the population
    population.append(elite_agent)
    elite_agent.fitness = elite_fitness

    return 0


def custom_starting_agent(filename, fitness):
    fittest = Agent(None)
    fittest.fitness = fitness
    fittest.moves = open(filename, "r").read()
    fittest.moves = fittest.moves.strip("[]")
    fittest.moves = [int(s) for s in fittest.moves.split(',')]
    fittest.moves_used = 1120
    return fittest

fittest = None
# fittest = custom_starting_agent("2227-fitness", 900)
for generation in range(generation_amount):
    population = create_population()
    if play_generation(fittest, population):
        break
    fittest = get_fittest(population)
    fitness_history.append(fittest.fitness)  # Collect the best fitness from this generation
    print("GENERATION {} HIGHEST FITNESS ACHIEVED: {}".format(generation,
                                                             fittest.fitness))
    winner = get_fittest(population)
    print("Player from generation {} won! Achieving a fitness of {} in {} moves!"
          .format(generation, winner.fitness, winner.moves_used))

    # Save animation as a GIF in the dedicated folder
    animation_filename = os.path.join(output_folder, f"generation_{generation}.gif")
    ani = display_animation(record_player(winner))
    ani.save(animation_filename, writer='pillow', fps=30)


winner = get_fittest(population)
print("Player from generation {} won! Achieving a fitness of {} in {} moves!"
      .format(generation, winner.fitness, winner.moves_used))
print("Compiling animation...")
ani = display_animation(record_player(winner))
ani
# Save fitness history to a plot
plt.figure()
plt.plot(fitness_history)
plt.xlabel('Generation')
plt.ylabel('Fitness')
plt.title('Fitness Progress')
plt.savefig(str(output_folder)+'/fitness_plot.pdf',format='pdf')

# Save animation as a GIF
ani.save(str(output_folder)+"/mario_flag_get-{}.gif".format(random.random()), writer='pillow', fps=30)


env.close()
plt.close('all')

