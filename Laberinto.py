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
    def __init__(self, maze, population_size=150, mutation_rate=0.6, crossover_rate=0.8, max_generations=1500):
        self.maze = maze
        self.population_size = population_size
        self.initial_mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.max_generations = max_generations
        self.population = self.initialize_population()
        self.best_fitness = -float('inf')
        self.stagnant_generations = 0
        self.max_stagnant_generations = 150

        # Ponderaciones para los factores de la función de fitness
        self.length_weight = 1.2
        self.obstacle_weight = 500
        self.turn_weight = 2.0
        self.repeat_penalty_weight = 300

        # Longitud esperada en función de la distancia euclidiana al objetivo
        self.expected_length = int(np.linalg.norm(np.array(maze.start) - np.array(maze.end)) * 1.3)

    def initialize_population(self):
        """Genera una población inicial de caminos moderadamente largos."""
        population = []
        for _ in range(self.population_size):
            path = [self.maze.start]
            for _ in range(random.randint(30, 40)):  # Longitud inicial moderada para exploración
                next_step = self.get_adjacent_step(path[-1], path)
                path.append(next_step)
            population.append(path)
        return population

    def get_adjacent_step(self, position, path):
        """Genera un paso adyacente aleatorio en todas las direcciones, evitando posiciones previas."""
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        random.shuffle(directions)
        for direction in directions:
            next_position = (position[0] + direction[0], position[1] + direction[1])
            if self.maze.is_free(next_position) and next_position not in path:
                return next_position
        return position  # Si no hay opciones válidas, permanece en la posición actual

    def fitness(self, path, generation):
        """Evalúa el camino considerando longitud, factibilidad, giros y acercamiento al objetivo."""
        # Penalización por distancia, con peso adaptativo que incrementa cerca del final
        distance_penalty = np.linalg.norm(np.array(path[-1]) - np.array(self.maze.end)) * (1 + generation / self.max_generations * 5)
        
        # Penalización de obstáculos
        obstacle_penalty = sum(1 for pos in path if pos in self.maze.obstacles) * self.obstacle_weight
        
        # Penalización de longitud, incentivando caminos cercanos a la longitud esperada
        length_penalty = abs(len(path) - self.expected_length) * self.length_weight

        # Penalización de giros para hacer el camino más fluido
        turns = sum(1 for i in range(1, len(path) - 1) if path[i-1][1] != path[i][1] and path[i][1] != path[i+1][1]) * self.turn_weight
        
        # Penalización de repetición de posiciones para evitar bucles
        repeat_penalty = (len(path) - len(set(path))) * self.repeat_penalty_weight

        # Fitness total
        fitness_value = -(distance_penalty + obstacle_penalty + length_penalty + turns + repeat_penalty)
        return fitness_value
    
    
