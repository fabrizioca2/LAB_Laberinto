import pygame
import random
import numpy as np

# --- Clase Maze ---
class Maze:
    def __init__(self, rows, cols, start, end, obstacles):
        self.rows = rows
        self.cols = cols
        self.start = start
        self.end = end
        self.obstacles = set(obstacles)

    def is_free(self, position):
        """Verifica si una posición en el laberinto es libre (sin obstáculos)."""
        return position not in self.obstacles and 0 <= position[0] < self.rows and 0 <= position[1] < self.cols

    def add_block_obstacle(self, top_left, width, height):
        """Agrega un obstáculo rectangular al laberinto."""
        for i in range(top_left[0], top_left[0] + height):
            for j in range(top_left[1], top_left[1] + width):
                if 0 <= i < self.rows and 0 <= j < self.cols:
                    self.obstacles.add((i, j))


# --- Clase GeneticAlgorithm ---
class GeneticAlgorithm:
    def __init__(self, maze, population_size=50, mutation_rate=0.1, crossover_rate=0.7, max_generations=2000):
        self.maze = maze
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.max_generations = max_generations
        self.population = self.initialize_population()
        self.best_fitness = -float('inf')
        self.stagnant_generations = 0
        self.max_stagnant_generations = 50  # Incrementado para permitir más iteraciones sin mejora

    def initialize_population(self):
        """Genera una población inicial de caminos contiguos largos desde el inicio."""
        population = []
        for _ in range(self.population_size):
            path = [self.maze.start]
            for _ in range(30):  # Longitud inicial del camino aumentada
                next_step = self.get_adjacent_step(path[-1], path)
                path.append(next_step)
            population.append(path)
        return population

    def get_adjacent_step(self, position, path):
        """Genera un paso adyacente aleatorio que no repita posiciones previas y sea contiguo."""
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        random.shuffle(directions)
        for direction in directions:
            next_position = (position[0] + direction[0], position[1] + direction[1])
            if self.maze.is_free(next_position) and next_position not in path:
                return next_position
        return position  # Si no hay opciones válidas, permanece en la posición actual
