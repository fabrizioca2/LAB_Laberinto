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

    def selection(self, generation):
        """Selecciona caminos con mejor fitness para la reproducción."""
        weighted_population = [(self.fitness(path, generation), path) for path in self.population]
        weighted_population.sort(reverse=True, key=lambda x: x[0])
        return [path for _, path in weighted_population[:int(self.population_size * 0.4)]]  # Mayor elitismo

    def crossover(self, parent1, parent2):
        """Realiza un cruce de un solo punto entre dos caminos, manteniendo contigüidad."""
        if random.random() > self.crossover_rate:
            return parent1
        
        crossover_point = random.randint(1, len(parent1) - 2)
        child_path = parent1[:crossover_point]

        # Añadir el resto de parent2 manteniendo adyacencia
        for i in range(crossover_point, len(parent2)):
            next_step = self.get_adjacent_step(child_path[-1], child_path)
            child_path.append(next_step)
        
        # Verificar contigüidad antes de retornar
        if self.is_path_contiguous(child_path):
            return child_path
        else:
            return parent1  # Si el cruce rompe la conectividad, usar el padre

    def mutate(self, path, generation):
        """Aplica una mutación adaptativa a un camino."""
        mutation_rate = self.initial_mutation_rate * (1 - (generation / self.max_generations))
        if random.random() < mutation_rate:
            mutate_index = random.randint(1, len(path) - 2)
            new_position = self.get_adjacent_step(path[mutate_index - 1], path)
            path[mutate_index] = new_position

    def is_path_contiguous(self, path):
        """Verifica si el camino es contiguo."""
        for i in range(1, len(path)):
            if abs(path[i][0] - path[i-1][0]) + abs(path[i][1] - path[i-1][1]) > 1:
                return False
        return True

    def evolve(self, generation):
        """Evoluciona la población mediante selección, cruce y mutación."""
        new_population = []
        best_path = max(self.population, key=lambda path: self.fitness(path, generation))
        new_population.append(best_path)  # Elitismo para mantener el mejor camino

        parents = self.selection(generation)
        for _ in range(self.population_size - 1):
            parent1, parent2 = random.sample(parents, 2)
            offspring = self.crossover(parent1, parent2)
            self.mutate(offspring, generation)
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
            best_path = self.evolve(generation)
            current_best_fitness = self.fitness(best_path, generation)
            print(f"Generación {generation}, Mejor Fitness: {current_best_fitness}")
            generation += 1
            
            # Visualiza el mejor camino de esta generación
            self.visualize(screen, best_path)

            # Verifica si el mejor camino llega al objetivo con longitud óptima o alcanza el objetivo
            if best_path[-1] == self.maze.end and self.is_path_contiguous(best_path):
                found_goal = True
                print(f"Camino óptimo encontrado en la generación {generation}. Algoritmo finalizado.")
                break

            # Verifica si el algoritmo está estancado
            if generation % 30 == 0:
                new_best_fitness = self.fitness(best_path, generation)
                if new_best_fitness == self.best_fitness:
                    self.stagnant_generations += 1
                    if self.stagnant_generations > self.max_stagnant_generations:
                        print("Algoritmo estancado. Finalizando búsqueda.")
                        break
                else:
                    self.best_fitness = new_best_fitness
                    self.stagnant_generations = 0

        if not found_goal:
            print("Algoritmo finalizado. No se encontró un camino óptimo en el límite de generaciones.")
        self.wait_for_exit(screen)

    def visualize(self, screen, path):
        """Visualiza el camino actual en pygame."""
        screen.fill((20, 20, 20))  # Fondo oscuro
        cell_size = 50

        # Dibujar obstáculos en gris oscuro
        for obstacle in self.maze.obstacles:
            pygame.draw.rect(screen, (128, 128, 128), (obstacle[1]*cell_size, obstacle[0]*cell_size, cell_size, cell_size))

        # Dibuja cada posición del camino en un gradiente de azul
        for i, pos in enumerate(path):
            intensity = 255 - int((i / len(path)) * 150)
            pygame.draw.rect(screen, (0, intensity, 255), (pos[1]*cell_size, pos[0]*cell_size, cell_size, cell_size))
            pygame.draw.rect(screen, (0, 0, 0), (pos[1]*cell_size, pos[0]*cell_size, cell_size, cell_size), 1)

        # Dibuja inicio en verde y el objetivo en rojo
        pygame.draw.rect(screen, (0, 255, 0), (self.maze.start[1]*cell_size, self.maze.start[0]*cell_size, cell_size, cell_size))
        pygame.draw.rect(screen, (255, 0, 0), (self.maze.end[1]*cell_size, self.maze.end[0]*cell_size, cell_size, cell_size))
        pygame.display.flip()
        pygame.time.delay(100)

    def wait_for_exit(self, screen):
        """Espera hasta que el usuario cierre la ventana."""
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
        pygame.quit()

# Ejecución Principal
pygame.init()
screen = pygame.display.set_mode((800, 800))

# Configuración del laberinto con obstáculos
maze = Maze(rows=15, cols=15, start=(0, 0), end=(14, 14), obstacles=set())
# Agregar obstáculos distribuidos
maze.add_block_obstacle((1, 2), 3, 2)
maze.add_block_obstacle((5, 5), 3, 5)
maze.add_block_obstacle((10, 2), 2, 3)
maze.add_block_obstacle((12, 8), 3, 4)

# Inicializar algoritmo genético y ejecutar
ga = GeneticAlgorithm(maze, max_generations=1500)
ga.run(screen)
