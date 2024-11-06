import pygame
import random
import numpy as np

# --- Clase Maze ---
class Maze:
    def _init_(self, rows, cols, start, end, obstacles):
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
    def _init_(self, maze, population_size=50, mutation_rate=0.2, crossover_rate=0.7, max_generations=1000):
        """Inicializa el algoritmo genético con un límite de generaciones."""
        self.maze = maze
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.max_generations = max_generations
        self.population = self.initialize_population()

   def initialize_population(self):
        """Genera una población inicial de caminos conectados desde el inicio."""
        population = []
        for _ in range(self.population_size):
            path = [self.maze.start]
            for _ in range(15):  # Longitud inicial del camino
                next_step = self.get_adjacent_step(path[-1], path)
                path.append(next_step)
            population.append(path)
        return population

    def get_adjacent_step(self, position, path):
        """Genera un paso adyacente desde la posición actual, asegurando continuidad y conectividad."""
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        random.shuffle(directions)
        for direction in directions:
            next_position = (position[0] + direction[0], position[1] + direction[1])
            if self.maze.is_free(next_position) and next_position not in path:
                return next_position
        return position  # Si no hay opciones válidas, permanece en la posición actual
    
    def is_path_contiguous(self, path):
        """Verifica que cada paso en el camino esté conectado al anterior."""
        for i in range(1, len(path)):
            x_diff = abs(path[i][0] - path[i-1][0])
            y_diff = abs(path[i][1] - path[i-1][1])
            if (x_diff + y_diff) != 1:  # Debe ser exactamente adyacente
                return False
        return True
    
    def fitness(self, path):
        """Evalúa un camino en función de la distancia al objetivo, obstáculos y penalización de repeticiones."""
        distance = np.linalg.norm(np.array(path[-1]) - np.array(self.maze.end))
        turns = sum(1 for i in range(1, len(path)-1) if path[i] != path[i-1] and path[i] != path[i+1])
        obstacle_penalty = sum(1 for pos in path if pos in self.maze.obstacles)
        repeat_penalty = len(path) - len(set(path))  # Penaliza caminos que repiten posiciones
        return -distance - (turns * 0.5) - (obstacle_penalty * 10) - (repeat_penalty * 2)
    
    def selection(self):
        """Selecciona los caminos con mayor fitness para cruzarlos en la siguiente generación."""
        weighted_population = [(self.fitness(path), path) for path in self.population]
        weighted_population.sort(reverse=True, key=lambda x: x[0])
        return [path for _, path in weighted_population[:self.population_size // 2]]
    
    def crossover(self, parent1, parent2):
        """Realiza un cruce entre dos caminos, asegurando que el hijo sea conectado y sin saltos."""
        if random.random() > self.crossover_rate:
            return parent1
        crossover_point = random.randint(1, len(parent1) - 2)
        child_path = parent1[:crossover_point]

        # Añade el resto de parent2 manteniendo adyacencia
        for i in range(crossover_point, len(parent2)):
            next_step = self.get_adjacent_step(child_path[-1], child_path)
            child_path.append(next_step)
        
        # Verificar contigüidad antes de retornar
        if self.is_path_contiguous(child_path):
            return child_path
        else:
            return parent1  # Si el cruce rompe la conectividad, usar el padre
        
    def mutate(self, path):
        """Aplica una mutación a un camino, moviendo una posición a una celda adyacente."""
        if random.random() < self.mutation_rate:
            mutate_index = random.randint(1, len(path) - 2)
            new_position = self.get_adjacent_step(path[mutate_index - 1], path)
            path[mutate_index] = new_position

    def evolve(self):
        """Evoluciona la población mediante selección, cruce y mutación."""
        new_population = []
        best_path = max(self.population, key=self.fitness)
        new_population.append(best_path)  # Elitismo para mantener el mejor camino

        parents = self.selection()
        for _ in range(self.population_size - 1):
            parent1, parent2 = random.sample(parents, 2)
            offspring = self.crossover(parent1, parent2)
            self.mutate(offspring)
            if self.is_path_contiguous(offspring):  # Añadir solo caminos contiguos
                new_population.append(offspring)
            else:
                new_population.append(parent1)  # En caso de falla, usar el padre como respaldo
        self.population = new_population
        return best_path

    def run(self, screen):
        """Ejecuta el algoritmo genético hasta encontrar el objetivo o alcanzar el límite de generaciones."""
        generation = 0
        best_path = None
        found_goal = False

        while generation < self.max_generations:
            best_path = self.evolve()
            print(f"Generación {generation}, Mejor Fitness: {self.fitness(best_path)}")
            generation += 1
            
            # Visualiza el mejor camino de esta generación
            self.visualize(screen, best_path)

            # Verifica si el mejor camino llega al objetivo
            if best_path[-1] == self.maze.end:
                found_goal = True
                break

        if found_goal:
            print("Camino óptimo encontrado antes de alcanzar el límite de generaciones.")
        else:
            print("Algoritmo finalizado. No se encontró un camino óptimo en el límite de generaciones.")

        self.wait_for_exit(screen)
