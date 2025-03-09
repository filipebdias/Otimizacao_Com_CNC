from ant_colony import AntColony
from differential_evolution import DifferentialEvolution
from genetic_algorithm import GeneticAlgorithm
from particle_swarm import ParticleSwarm

def main():
    # Define sheet dimensions
    sheet_width = 200
    sheet_height = 100

   # Define available parts (recortes_disponiveis) as a JSON-like structure.
    recortes_disponiveis = [
    {"tipo": "retangular", "largura": 30, "altura": 30, "x": 1, "y": 1, "rotacao": 0},
    {"tipo": "retangular", "largura": 30, "altura": 30, "x": 35, "y": 1, "rotacao": 0},
    {"tipo": "retangular", "largura": 30, "altura": 30, "x": 1, "y": 35, "rotacao": 0},
    {"tipo": "circular", "r": 15, "x": 70, "y": 1},
    {"tipo": "circular", "r": 15, "x": 95, "y": 1}
]


    # Instantiate and run Particle Swarm Optimization.
    ps_optimizer = ParticleSwarm(num_particles=50, num_iterations=100, dim=len(recortes_disponiveis),
                                 sheet_width=sheet_width, sheet_height=sheet_height, recortes_disponiveis=recortes_disponiveis)
    print("Running Particle Swarm Optimization...")
    ps_optimized_layout = ps_optimizer.optimize_and_display()

    # Display optimized layout
    print("Particle Swarm Optimized Layout:")
    for item in ps_optimized_layout or []:
        print(item)

if __name__ == "__main__":
    main()
