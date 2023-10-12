from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import RIGHT_ONLY
import random
import numpy as np

# Define the neural network class
class NeuralNetwork:
    def __init__(self, num_inputs, num_outputs):
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.weights = np.random.randn(num_inputs, num_outputs)

    def predict(self, inputs):
        if len(inputs) != self.num_inputs:
            raise ValueError("Input size mismatch")
        return np.dot(inputs, self.weights)

# Create the Super Mario environment
env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')
env = JoypadSpace(env, RIGHT_ONLY)

# NEAT Parameters
population_size = 1
max_generations = 1000
mutation_rate = 0.1

# Define the number of available actions
num_actions = len(RIGHT_ONLY)

def run_episode(neural_network):
    total_reward = 0
    observation = env.reset()
    done = False
    while not done:
        action = np.argmax(neural_network.predict(observation.flatten()))  # Flatten the observation
        observation, reward, done, _ = env.step(action)
        total_reward += reward
    return total_reward

# Initialize the population with random neural networks
population = [NeuralNetwork(env.observation_space.shape[0] * env.observation_space.shape[1] * env.observation_space.shape[2], num_actions) for _ in range(population_size)]

for generation in range(max_generations):
    # Evaluate each neural network in the population
    fitness_scores = []
    for i, neural_network in enumerate(population):
        fitness = run_episode(neural_network)
        fitness_scores.append(fitness)
        print(f"Generation {generation}, Agent {i}: Fitness = {fitness}")

    # Select the top performers based on fitness
    num_top_performers = int(0.2 * population_size)  # 20% of the population
    top_performers_indices = np.argsort(fitness_scores)[-num_top_performers:]

    # Create a new population with the top performers
    new_population = [population[i] for i in top_performers_indices]

    # Mutate and crossover to create the rest of the population
    while len(new_population) < population_size:
        parent1 = random.choice(new_population)
        parent2 = random.choice(new_population)
        child = NeuralNetwork(env.observation_space.shape[0] * env.observation_space.shape[1] * env.observation_space.shape[2], num_actions)

        for i in range(child.num_inputs):
            for j in range(child.num_outputs):
                if random.uniform(0, 1) < mutation_rate:
                    child.weights[i, j] = random.uniform(-1, 1)
                else:
                    child.weights[i, j] = random.choice([parent1.weights[i, j], parent2.weights[i, j]])

        new_population.append(child)

    population = new_population

# Find the best neural network after evolution
best_neural_network = max(population, key=lambda neural_network: run_episode(neural_network))

# Play the game with the best neural network
final_reward = run_episode(best_neural_network)
print("Best neural network achieved a reward of:", final_reward)

# Close the environment
env.close()
