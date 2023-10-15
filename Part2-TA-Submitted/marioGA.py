
#! /usr/bin/python3
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import *
import matplotlib.pyplot as plt
from matplotlib import animation, rc
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
import random, sys
env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')
env = JoypadSpace(env, RIGHT_ONLY)
#Animation
save_animation = 1 # Saves animation to plot
frames_skipped = 1 # How many frames to skip when creating animation

#Generation
generation_amount = 1000

#Population
population_amount = 10 # Amount of agents per generation

#Agent
moves_amount = 5000 # Max amount of moves agent can perform
moves_to_check = 30 # How many moves without increase in fitness until termination
mutation_rate = 1.001 # Chance of any move changing to a random move
moves_mutable = 0.2 # How many percent of moves are mutable (starting from end)
                    # Regardless of percent, a minimum of (moves_to_check*2) moves
                    # are mutable.


t = 0
def print_info(info, reward):
    global t 
    t += 1
    if not t % 100:
        print(info, reward)
def handle_frame(step, images):
    if not step % frames_skipped:
        image = plt.imshow(env.render(mode='rgb_array'))
        images.append([image])
fig = plt.figure()
def display_animation(images):
    anim = animation.ArtistAnimation(fig, images,
                                     interval=15*frames_skipped, blit=True)
    rc('animation', html='jshtml')
    return(anim)
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
def check_fitness(player, fitness, past_fitness):
    if player.fitness < fitness:
        player.fitness = fitness
    if len(past_fitness) < moves_to_check:
        past_fitness.append(fitness)
    else:
        past_fitness.pop(0)
        past_fitness.append(fitness)
        for i in range(moves_to_check):
            if past_fitness[i] > past_fitness[0]:
                break
            #Kills player if no progress in fitness for moves_to_check moves
            if i == moves_to_check - 1:
                return True
def mutate_moves(player):
    start_index = int(player.moves_used * (1-moves_mutable))
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
    mutable_amount = current_fittest.moves_used * moves_mutable # new line
    if fittest and fittest.fitness == current_fittest.fitness:
        print("Increasing mutation chance by 0.005")
        mutation_rate += 0.005
        print("Mutation chance is now {}%".format(mutation_rate*100))
    else:
        print("Fitness increased succesfully, restore mutation chance to 0.01")
        mutation_rate = 0.01
        mutable_amount = current_fittest.moves_used * moves_mutable

    if mutable_amount < moves_to_check * 2: 
        mutable_amount = moves_to_check * 2  
        

    print("Mutating last {}% of moves, equal to {} moves".format(moves_mutable*100,
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
        state, reward, done,info = env.step(player.moves[move])
        # state, reward, done,info = env.step(player.moves[move])
        if done:
            break
        done = check_fitness(player, info['x_pos'], past_fitness)
        handle_frame(move, images)
    return images
        
def play_generation(fittest, population):
    for i in range(population_amount):
        player = Agent(fittest)
        observation = env.reset()
        done = False
        past_fitness = []
        agent_get_moves(player, population)
        for move in range(len(player.moves)):
            if done:
                break
            state, reward, done,info = env.step(player.moves[move])
            # state, reward, done,info = env.step(player.moves[move])
            if done:
                break
            done = check_fitness(player, info['x_pos'], past_fitness)
            env.render()
        player.moves_used = move
        population.append(player)
        print("Player {} achieved a fitness of {} in {} moves".format(i,
                                                               player.fitness, move))
        if info['flag_get']:
            return 1
    print_mutation_info(fittest, population)
    return 0


def custom_starting_agent(filename, fitness):
    fittest = Agent(None)
    fittest.fitness = fitness
    fittest.moves = open(filename, "r").read()
    fittest.moves = fittest.moves.strip("[]")
    fittest.moves = [int(s) for s in fittest.moves.split(',')]
    fittest.moves_used = 1120
    return (fittest)



fittest = None
#fittest = custom_starting_agent("2227-fitness", 900)
for generation in range(generation_amount):
    population = []
    if play_generation(fittest, population):
        break
    fittest = get_fittest(population)
    print("GENERATION {} HIGHEST FITNESS ACHIEVED: {}".format(generation,
                                                             fittest.fitness))


winner = get_fittest(population)
print("Player from generation {} won! Achieving a fitness of {} in {} moves!"
          .format(generation, winner.fitness, winner.moves_used))
print("Compiling animation...")
ani = display_animation(record_player(winner))
ani


Writer = animation.FFMpegFileWriter
writer = Writer(fps=20, metadata=dict(artist='Me'), bitrate=-1)
ani.save("mario_flag_get-{}.mp4".format(random.random()), writer = writer)


env.close()
plt.close('all')





