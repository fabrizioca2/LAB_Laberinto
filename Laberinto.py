import pygame
import random
import numpy as np

# --- Clase Maze ---
class Maze:
    def __init__(self, rows, cols, start, end, obstacles):
        """Inicializa el laberinto con sus dimensiones, punto de inicio, punto final y obstáculos."""
        self.rows = rows
        self.cols = cols
        self.start = start
        self.end = end
        self.obstacles = set(obstacles)

    def is_free(self, position):
        """Verifica si una posición en el laberinto es libre (sin obstáculos)."""
        return position not in self.obstacles and 0 <= position[0] < self.rows and 0 <= position[1] < self.cols
    
# --- Clase GeneticAlgorithm ---
class GeneticAlgorithm:
    def __init__(self, maze, population_size=100, mutation_rate=0.2, crossover_rate=0.8, max_generations=2000):
        """Inicializa el algoritmo genético con un límite de generaciones."""
        self.maze = maze
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.max_generations = max_generations
        self.population = self.initialize_population()
    
        def initialize_population(self):
             """Genera una población inicial de caminos contiguos desde el inicio."""
                population = []
                for _ in range(self.population_size):
                    path = [self.maze.start]
                    for _ in range(20):  # Aumenta la longitud inicial para explorar más áreas
                        next_step = self.get_continuous_step(path[-1])
                        path.append(next_step)
                    population.append(path)
                return population
