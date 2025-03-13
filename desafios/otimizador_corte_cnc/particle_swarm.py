from common.layout_display import LayoutDisplayMixin
import random

class ParticleSwarm(LayoutDisplayMixin):
    def __init__(self, num_particles, num_iterations, dim, sheet_width, sheet_height, recortes_disponiveis):
        """
        Initializes the Particle Swarm optimizer.
        :param num_particles: Number of particles.
        :param num_iterations: Number of iterations to run.
        :param dim: Dimensionality of the problem.
        :param sheet_width: Width of the cutting sheet.
        :param sheet_height: Height of the cutting sheet.
        :param recortes_disponiveis: List of available parts (JSON structure).
        """
        self.num_particles = num_particles
        self.num_iterations = num_iterations
        self.dim = dim
        self.sheet_width = sheet_width
        self.sheet_height = sheet_height
        self.initial_layout = recortes_disponiveis
        self.optimized_layout = None
        self.best_global_position = None
        self.best_global_score = float('inf')
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
                "best_position": position,
                "best_score": score
            })
        self.update_global_best()
        print(f"Initialized {self.num_particles} particles.")

    def random_position(self):
        return [{"x": random.uniform(0, self.sheet_width), "y": random.uniform(0, self.sheet_height)} for _ in self.initial_layout]

    def evaluate(self, position):
     total_area_used = 0

     for i, pos in enumerate(position):
        # Pegando o recorte correspondente
        recorte = self.initial_layout[i]
        x, y = pos["x"], pos["y"]

        # Verifica se o recorte cabe dentro da chapa
        if 0 <= x + recorte["x"] <= self.sheet_width and 0 <= y + recorte["y"] <= self.sheet_height:
            total_area_used += recorte["x"] * recorte["y"]

     sheet_area = self.sheet_width * self.sheet_height
     wasted_space = sheet_area - total_area_used

     return -wasted_space  # Negativo para que o algoritmo tente maximizar o uso da chapa


    def update_velocity(self, particle):
        w = 0.5  # Inércia
        c1 = 1.5  # Componente cognitivo
        c2 = 1.5  # Componente social

        for i in range(self.dim):
            r1, r2 = random.random(), random.random()
            cognitive_velocity = c1 * r1 * (particle["best_position"][i]["x"] - particle["position"][i]["x"])
            social_velocity = c2 * r2 * (self.best_global_position[i]["x"] - particle["position"][i]["x"])
            particle["velocity"][i]["x"] = w * particle["velocity"][i]["x"] + cognitive_velocity + social_velocity

            cognitive_velocity_y = c1 * r1 * (particle["best_position"][i]["y"] - particle["position"][i]["y"])
            social_velocity_y = c2 * r2 * (self.best_global_position[i]["y"] - particle["position"][i]["y"])
            particle["velocity"][i]["y"] = w * particle["velocity"][i]["y"] + cognitive_velocity_y + social_velocity_y

    def update_position(self, particle):
     for i in range(self.dim):
        # Atualizando as posições
        particle["position"][i]["x"] += particle["velocity"][i]["x"]
        particle["position"][i]["y"] += particle["velocity"][i]["y"]

        # Garantir que a partícula fique dentro dos limites
        particle["position"][i]["x"] = max(0, min(particle["position"][i]["x"], self.sheet_width))
        particle["position"][i]["y"] = max(0, min(particle["position"][i]["y"], self.sheet_height))

        # Verificar se o recorte ainda cabe dentro da chapa
        recorte = self.initial_layout[i]
        x, y = particle["position"][i]["x"], particle["position"][i]["y"]

        # Verificar se a posição do recorte ultrapassa os limites
        if x + recorte["x"] > self.sheet_width:
            particle["position"][i]["x"] = self.sheet_width - recorte["x"]
        
        if y + recorte["y"] > self.sheet_height:
            particle["position"][i]["y"] = self.sheet_height - recorte["y"]

        # Garantir que a posição não seja negativa (caso algum erro ocorra)
        particle["position"][i]["x"] = max(0, particle["position"][i]["x"])
        particle["position"][i]["y"] = max(0, particle["position"][i]["y"])

    def update_global_best(self):
        for particle in self.particles:
            if particle["score"] < self.best_global_score:
                self.best_global_position = [dict(pos) for pos in particle["position"]]
                self.best_global_score = particle["score"]

    def run(self):
        self.initialize_particles()
        for iteration in range(self.num_iterations):
            for particle in self.particles:
                particle["score"] = self.evaluate(particle["position"])
                
                if particle["score"] < particle["best_score"]:
                    particle["best_position"] = particle["position"]
                    particle["best_score"] = particle["score"]

            self.update_global_best()

            for particle in self.particles:
                self.update_velocity(particle)
                self.update_position(particle)

            print(f"Iteration {iteration+1}/{self.num_iterations} - Best Score: {self.best_global_score}")

        return self.best_global_position

    def optimize_and_display(self):
        """
        Displays the initial layout, runs the optimization, and then displays the optimized layout.
        """
        # Display initial layout
        self.display_layout(self.initial_layout, title="Initial Layout - Particle Swarm")

        # Run the optimization (this should update self.optimized_layout)
        self.optimized_layout = self.run()

        # Display optimized layout
        self.display_layout(self.optimized_layout, title="Optimized Layout - Particle Swarm")
        return self.optimized_layout
    
    def recortes_disponiveis(self, particle):
       total_area_used = 0
    
       for i in range(self.dim):
        # Obtém a posição do recorte
        x, y = particle["position"][i]["x"], particle["position"][i]["y"]
        
        # Obtém as dimensões do recorte (largura e altura)
        recorte = self.initial_layout[i]
        largura, altura = recorte["largura"], recorte["altura"]

        # Verifica se o recorte cabe na chapa
        if 0 <= x + largura <= self.sheet_width and 0 <= y + altura <= self.sheet_height:
            total_area_used += largura * altura

       return total_area_used
