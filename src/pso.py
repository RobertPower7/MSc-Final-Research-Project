import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt


class Particle:
    def __init__(self, model, position_init_range, velocity_init_range):
        # Number of parameters in the model
        self.dim = sum(p.numel() for p in model.parameters())

        # Position & velocity initialization
        self.position = np.random.uniform(position_init_range[0], position_init_range[1], self.dim)
        self.velocity = np.random.uniform(velocity_init_range[0], velocity_init_range[1], self.dim)

        # Personal best position and the associated error (particle fitness)
        self.best_position = self.position.copy()
        self.best_error = float('inf')


class PSO:
    def __init__(self, model, boundary_interval, position_init_range, velocity_init_range, swarm_size=50,
                 inertia=0.729844, c1=1.496180, c2=1.496180):
        self.model = model
        self.dim = sum(p.numel() for p in model.parameters())

        self.boundary_interval = boundary_interval
        self.position_init_range = position_init_range
        self.velocity_init_range = velocity_init_range

        # Initialize particles according to model parameters
        self.particles = [Particle(model, position_init_range, velocity_init_range) for _ in range(swarm_size)]

        self.inertia = inertia
        self.c1 = c1
        self.c2 = c2

        # Initialize global best position and the associated error (particle fitness)
        self.global_best_error = float('inf')
        self.global_best_position = None

    def update_velocity_position(self, particle, global_best):
        r1, r2 = np.random.rand(2, particle.position.size)
        cognitive = self.c1 * r1 * (particle.best_position - particle.position)
        social = self.c2 * r2 * (global_best - particle.position)

        particle.velocity = self.inertia * particle.velocity + cognitive + social
        particle.velocity = np.clip(particle.velocity, self.velocity_init_range[0], self.velocity_init_range[1])

        particle.position += particle.velocity
        particle.position = np.clip(particle.position, self.boundary_interval[0], self.boundary_interval[1])

    def optimize(self, train_loader, test_loader, epochs=10, iterations=500, stop_by='epochs', activation='sigmoid'):
        # Initialize lists to record train and test errors
        train_errors = []
        test_errors = []

        # Set stopping criterion
        stopping_criterion = epochs if stop_by == 'epochs' else iterations
        if stop_by not in ['epochs', 'iterations']:
            raise ValueError("stop_by must be 'epochs' or 'iterations'")

        # Use tqdm to track progress
        with tqdm(total=stopping_criterion, desc="Optimization Progress") as pbar:
            if stop_by == 'epochs':
                for epoch in range(epochs):
                    for particle in self.particles:
                        # Set model weights from particle position
                        self.set_model_weights_from_particle(particle.position)
                        # Calculate training error
                        train_error = self.evaluate(train_loader, activation)

                        # Update particle's best position and error
                        if train_error < particle.best_error:
                            particle.best_error = train_error
                            particle.best_position = particle.position.copy()

                        # Update global best if current error is lower
                        if train_error < self.global_best_error:
                            self.global_best_error = train_error
                            self.global_best_position = particle.position.copy()

                    # Update velocities and positions for each particle
                    for particle in self.particles:
                        self.update_velocity_position(particle, self.global_best_position)

                    # Record train and test errors
                    train_errors.append(train_error)
                    test_error = self.evaluate(test_loader, activation)
                    test_errors.append(test_error)

                    # Update tqdm progress bar with current train and test errors
                    pbar.set_postfix({"Train Error": train_error, "Test Error": test_error})
                    pbar.update(1)

            elif stop_by == 'iterations':
                current_iteration = 0
                while current_iteration < iterations:
                    for particle in self.particles:
                        # Set model weights from particle position
                        self.set_model_weights_from_particle(particle.position)
                        # Calculate training error
                        train_error = self.evaluate(train_loader, activation)

                        # Update particle's best position and error
                        if train_error < particle.best_error:
                            particle.best_error = train_error
                            particle.best_position = particle.position.copy()

                        # Update global best if current error is lower
                        if train_error < self.global_best_error:
                            self.global_best_error = train_error
                            self.global_best_position = particle.position.copy()

                    # Update velocities and positions for each particle
                    for particle in self.particles:
                        self.update_velocity_position(particle, self.global_best_position)

                    # Record train and test errors
                    train_errors.append(train_error)
                    test_error = self.evaluate(test_loader, activation)
                    test_errors.append(test_error)

                    # Update tqdm progress bar with current train and test errors
                    pbar.set_postfix(
                        {"Iteration": current_iteration + 1, "Train Error": train_error, "Test Error": test_error})
                    pbar.update(1)
                    current_iteration += 1

        # Plot train and test errors after optimization
        plt.figure(figsize=(10, 6))
        plt.plot(train_errors, label='Train Error')
        plt.plot(test_errors, label='Test Error')
        plt.xlabel('Epoch' if stop_by == 'epochs' else 'Iteration')
        plt.ylabel('Error')
        plt.title('Train and Test Error Over Time')
        plt.legend()
        plt.show()

    def set_model_weights_from_particle(self, particle_position):
        param_vector = torch.tensor(particle_position, dtype=torch.float32)
        idx = 0
        for param in self.model.parameters():
            numel = param.numel()
            param.data = param_vector[idx: idx + numel].view_as(param).data
            idx += numel

    def evaluate(self, data_loader, activation):
        self.model.eval()
        total_error = 0
        with torch.no_grad():
            for data, target in data_loader:
                output = self.model(data, activation=activation)
                loss = torch.nn.functional.cross_entropy(output, target)
                total_error += loss.item()
        return total_error / len(data_loader)
