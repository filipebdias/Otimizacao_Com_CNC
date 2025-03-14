from common.layout_display import LayoutDisplayMixin
import random
import time
import matplotlib.pyplot as plt

class ParticleSwarm(LayoutDisplayMixin):
    def __init__(self, num_particles, num_iterations, dim, sheet_width, sheet_height, recortes_disponiveis):
        self.num_particles = num_particles
        self.num_iterations = num_iterations
        self.dim = dim
        self.sheet_width = sheet_width
        self.sheet_height = sheet_height
        self.initial_layout = recortes_disponiveis
        self.optimized_layout = None
        self.best_global_position = None
        self.best_global_score = float('inf')
        self.total_energy_consumption = 0
        self.total_cutting_time = 0
        self.execution_time = 0
        self.best_scores = []  # Armazena o melhor score de cada iteração
        self.total_costs = []  # Armazena o custo total de cada iteração
        print("Particle Swarm Optimization Initialized.")

    def initialize_particles(self):
        self.particles = []
        for _ in range(self.num_particles):
            position = self.random_position()
            velocity = [{"x": random.uniform(-1, 1), "y": random.uniform(-1, 1)} for _ in range(self.dim)]
            score = self.evaluate(position)
            self.particles.append({
                "position": position,
                "velocity": velocity,
                "score": score,
                "best_position": position[:],
                "best_score": score
            })
        self.update_global_best()
        print(f"Initialized {self.num_particles} particles.")

    def random_position(self):
        return [{"x": random.uniform(0, self.sheet_width), "y": random.uniform(0, self.sheet_height)} for _ in self.initial_layout]

    def evaluate(self, position):
        total_area_used = 0
        penalty = 0
        self.total_energy_consumption = 0
        self.total_cutting_time = 0

        for i, pos in enumerate(position):
            recorte = self.initial_layout[i]
            x, y = pos["x"], pos["y"]

            # Verifica o tipo de recorte
            if recorte["tipo"] == "retangular":
                width = recorte["largura"]
                height = recorte["altura"]
            elif recorte["tipo"] == "circular":
                width = height = 2 * recorte["r"]  # Diâmetro do círculo
            elif recorte["tipo"] == "diamante":
                width = recorte["largura"]
                height = recorte["altura"]
            else:
                raise ValueError(f"Tipo de recorte desconhecido: {recorte['tipo']}")

            # Verifica se o recorte está dentro dos limites da chapa
            if 0 <= x and x + width <= self.sheet_width and 0 <= y and y + height <= self.sheet_height:
                if recorte["tipo"] == "retangular" or recorte["tipo"] == "diamante":
                    total_area_used += width * height
                elif recorte["tipo"] == "circular":
                    total_area_used += 3.1416 * (recorte["r"] ** 2)  # Área do círculo (πr²)
                # Exemplo: tempo de corte proporcional ao perímetro do recorte
                self.total_cutting_time += 2 * (width + height)
                # Exemplo: consumo de energia proporcional à área do recorte
                self.total_energy_consumption += width * height * 0.1
            else:
                penalty += 100  # Penalidade reduzida por recorte fora dos limites

            # Verificação de colisões
            for j in range(i):
                recorte_anterior = self.initial_layout[j]
                pos_anterior = position[j]
                if self.check_collision(pos, recorte, pos_anterior, recorte_anterior):
                    penalty += 50  # Penalidade reduzida por colisão

        wasted_space = (self.sheet_width * self.sheet_height) - total_area_used
        # Minimizar espaço desperdiçado, tempo de corte, consumo de energia e penalidades
        return wasted_space + self.total_cutting_time + self.total_energy_consumption + penalty

    def check_collision(self, pos1, recorte1, pos2, recorte2):
        x1, y1 = pos1["x"], pos1["y"]
        x2, y2 = pos2["x"], pos2["y"]

        # Verifica o tipo de recorte
        if recorte1["tipo"] == "retangular" or recorte1["tipo"] == "diamante":
            width1 = recorte1["largura"]
            height1 = recorte1["altura"]
        elif recorte1["tipo"] == "circular":
            width1 = height1 = 2 * recorte1["r"]
        else:
            raise ValueError(f"Tipo de recorte desconhecido: {recorte1['tipo']}")

        if recorte2["tipo"] == "retangular" or recorte2["tipo"] == "diamante":
            width2 = recorte2["largura"]
            height2 = recorte2["altura"]
        elif recorte2["tipo"] == "circular":
            width2 = height2 = 2 * recorte2["r"]
        else:
            raise ValueError(f"Tipo de recorte desconhecido: {recorte2['tipo']}")

        if (x1 < x2 + width2 and
            x1 + width1 > x2 and
            y1 < y2 + height2 and
            y1 + height1 > y2):
            return True
        return False

    def update_velocity(self, particle):
        w = 0.7  # Inércia aumentada para melhorar a exploração
        c1 = 1.2  # Aceleração cognitiva
        c2 = 1.2  # Aceleração social
        for i in range(self.dim):
            r1, r2 = random.random(), random.random()
            cognitive = c1 * r1 * (particle["best_position"][i]["x"] - particle["position"][i]["x"])
            social = c2 * r2 * (self.best_global_position[i]["x"] - particle["position"][i]["x"])
            particle["velocity"][i]["x"] = w * particle["velocity"][i]["x"] + cognitive + social
            cognitive_y = c1 * r1 * (particle["best_position"][i]["y"] - particle["position"][i]["y"])
            social_y = c2 * r2 * (self.best_global_position[i]["y"] - particle["position"][i]["y"])
            particle["velocity"][i]["y"] = w * particle["velocity"][i]["y"] + cognitive_y + social_y

    def update_position(self, particle):
        for i in range(self.dim):
            particle["position"][i]["x"] = max(0, min(particle["position"][i]["x"] + particle["velocity"][i]["x"], self.sheet_width))
            particle["position"][i]["y"] = max(0, min(particle["position"][i]["y"] + particle["velocity"][i]["y"], self.sheet_height))

    def mutate_particle(self, particle, mutation_rate=0.1):
        for i in range(self.dim):
            if random.random() < mutation_rate:
                particle["position"][i]["x"] = random.uniform(0, self.sheet_width)
                particle["position"][i]["y"] = random.uniform(0, self.sheet_height)

    def update_global_best(self):
        for particle in self.particles:
            if particle["score"] < self.best_global_score:
                self.best_global_position = [dict(pos) for pos in particle["position"]]
                self.best_global_score = particle["score"]

    def run(self):
        start_time = time.time()
        self.initialize_particles()
        for iteration in range(self.num_iterations):
            for particle in self.particles:
                particle["score"] = self.evaluate(particle["position"])
                if particle["score"] < particle["best_score"]:
                    particle["best_position"] = particle["position"][:]
                    particle["best_score"] = particle["score"]
            self.update_global_best()
            for particle in self.particles:
                self.update_velocity(particle)
                self.update_position(particle)
                self.mutate_particle(particle)  # Adiciona mutação para diversidade
            
            # Armazena o melhor score e o custo total da iteração
            self.best_scores.append(self.best_global_score)
            self.total_costs.append(self.calculate_total_cost(self.best_global_position))
            
            print(f"Iteration {iteration+1}/{self.num_iterations} - Best Score: {self.best_global_score}")
        
        end_time = time.time()
        self.execution_time = end_time - start_time
        print(f"Total execution time: {self.execution_time:.2f} seconds")
        return self.best_global_position

    def calculate_total_cost(self, position):
        wasted_space = self.evaluate(position)
        material_cost_per_unit = 10  # Custo por unidade de área de material
        energy_cost_per_unit = 0.5  # Custo por unidade de energia
        time_cost_per_unit = 2  # Custo por unidade de tempo de máquina

        total_material_cost = wasted_space * material_cost_per_unit
        total_energy_cost = self.total_energy_consumption * energy_cost_per_unit
        total_time_cost = self.total_cutting_time * time_cost_per_unit

        return total_material_cost + total_energy_cost + total_time_cost

    def plot_evolution(self):
        plt.figure(figsize=(12, 6))
        
        # Plot do melhor score
        plt.subplot(1, 2, 1)
        plt.plot(self.best_scores, label="Best Score")
        plt.xlabel("Iteration")
        plt.ylabel("Best Score")
        plt.title("Evolution of Best Score")
        plt.legend()
        
        # Plot do custo total
        plt.subplot(1, 2, 2)
        plt.plot(self.total_costs, label="Total Cost", color="orange")
        plt.xlabel("Iteration")
        plt.ylabel("Total Cost")
        plt.title("Evolution of Total Cost")
        plt.legend()
        
        plt.tight_layout()
        plt.show()

    def display_results(self, optimized_layout):
        print("\nOptimized Layout:")
        for item in optimized_layout:
            print(item)
        
        economic_impact = self.calculate_economic_impact(optimized_layout)
        print("\nEconomic Impact:")
        print(f"Total Material Cost: {economic_impact['total_material_cost']}")
        print(f"Total Energy Cost: {economic_impact['total_energy_cost']}")
        print(f"Total Time Cost: {economic_impact['total_time_cost']}")
        print(f"Total Cost: {economic_impact['total_cost']}")

        best_solution = self.get_best_solution()
        print("\nBest Solution:")
        print(f"Best Position: {best_solution['best_position']}")
        print(f"Best Score: {best_solution['best_score']}")
        print(f"Execution Time: {best_solution['execution_time']:.2f} seconds")

    def calculate_economic_impact(self, optimized_layout):
        wasted_space = self.evaluate(optimized_layout)
        material_cost_per_unit = 10  # Custo por unidade de área de material
        energy_cost_per_unit = 0.5  # Custo por unidade de energia
        time_cost_per_unit = 2  # Custo por unidade de tempo de máquina

        total_material_cost = wasted_space * material_cost_per_unit
        total_energy_cost = self.total_energy_consumption * energy_cost_per_unit
        total_time_cost = self.total_cutting_time * time_cost_per_unit

        total_cost = total_material_cost + total_energy_cost + total_time_cost
        return {
            "total_material_cost": total_material_cost,
            "total_energy_cost": total_energy_cost,
            "total_time_cost": total_time_cost,
            "total_cost": total_cost
        }

    def get_best_solution(self):
        return {
            "best_position": self.best_global_position,
            "best_score": self.best_global_score,
            "execution_time": self.execution_time
        }

    def optimize_and_display(self):
        self.display_layout(self.initial_layout, title="Initial Layout - Particle Swarm")
        self.optimized_layout = self.run()
        self.display_layout(self.optimized_layout, title="Optimized Layout - Particle Swarm")
        self.display_results(self.optimized_layout)
        self.plot_evolution()
        return self.optimized_layout