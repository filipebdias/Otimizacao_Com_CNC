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
        # Initialize particle positions and velocities.
        self.particles = []
        for _ in range(self.num_particles):
            position = self.random_position()
            velocity = [random.uniform(-1, 1) for _ in range(self.dim)]
            score = self.evaluate(position)
            self.particles.append({
                "position": position,
                "velocity": velocity,
                "score": score,
                "best_position": position,
                "best_score": score
            })
        print(f"Initialized {self.num_particles} particles.")

    def random_position(self):
        # Randomly initialize the position of a particle.
        return [
            {"x": random.uniform(0, self.sheet_width), "y": random.uniform(0, self.sheet_height)}
            for _ in self.initial_layout
        ]

    def evaluate(self, position):
        # Evaluation function to calculate the fitness score of a particle's position.
        # Here, it can be a simple function like calculating how well the parts fit into the sheet.
        # Example: Minimize wasted space
        total_area_used = sum([part["largura"] * part["altura"] for part in position])
        sheet_area = self.sheet_width * self.sheet_height
        wasted_space = sheet_area - total_area_used
        return wasted_space

    def update_velocity(self, particle):
        # Update the velocity of each particle based on personal and global best positions.
        w = 0.5  # inertia weight
        c1 = 1.5  # cognitive coefficient
        c2 = 1.5  # social coefficient

        for i in range(self.dim):
            r1 = random.random()
            r2 = random.random()

            cognitive_velocity = c1 * r1 * (particle["best_position"][i]["x"] - particle["position"][i]["x"])
            social_velocity = c2 * r2 * (self.best_global_position[i]["x"] - particle["position"][i]["x"])

            particle["velocity"][i] = w * particle["velocity"][i] + cognitive_velocity + social_velocity

    def update_position(self, particle):
        # Update the position of each particle using the updated velocity.
        for i in range(self.dim):
            particle["position"][i]["x"] += particle["velocity"][i]

            # Ensure the particle stays within the sheet boundaries
            particle["position"][i]["x"] = max(0, min(particle["position"][i]["x"], self.sheet_width))
            particle["position"][i]["y"] = max(0, min(particle["position"][i]["y"], self.sheet_height))

    def get_best_solution(self):
        # Return the best solution found.
        return self.best_global_position, self.best_global_score

    def run(self):
        """
        Executes the main loop of the Particle Swarm algorithm.
        This method should return the optimized layout (JSON structure).
        # Main PSO loop:
        # 1. Evaluate particles.
        # 2. Update personal and global bests.
        # 3. Update velocities.
        # 4. Update positions.
        """
        # Initialize particles
        self.initialize_particles()

        # Main loop
        for iteration in range(self.num_iterations):
            for particle in self.particles:
                # Evaluate particle
                particle["score"] = self.evaluate(particle["position"])

                # Update personal best
                if particle["score"] < particle["best_score"]:
                    particle["best_position"] = particle["position"]
                    particle["best_score"] = particle["score"]

                # Update global best
                if particle["score"] < self.best_global_score:
                    self.best_global_position = particle["position"]
                    self.best_global_score = particle["score"]

            # Update velocities and positions
            for particle in self.particles:
                self.update_velocity(particle)
                self.update_position(particle)

            # Print progress
            print(f"Iteration {iteration+1}/{self.num_iterations} - Best Score: {self.best_global_score}")

        # Return the best solution found after all iterations
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
