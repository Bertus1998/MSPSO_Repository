import math
import random

import numpy as np

ACCELERATION_MIN = 0.4
ACCELERATION_MAX = 2


def _generate_accelerate_coefs(size, phi):
    return np.random.uniform(low=0, high=phi, size=size)


def linear_interpolation(start, end, coeff):
    return (1 - coeff) * start + coeff * end


def square_interpolation(start, end, coeff):
    return end ** coeff + start ** (1 - coeff)


def const_function(start, end, coeff):
    return start + end / 2


# od 0 do 1,
def sphere_func(x: np.array):
    return np.sum(x ** 2)


# Rastrigin
def f5_func(x: np.array):
    return np.sum(x ** 2 - 10 * np.cos(math.pi * x) + 10)


def f2_func(x: np.array):
    indexes = np.arange(1, x.size + 1, 1)
    return np.sum((x - indexes) ** 2)


def griewank_func(x: np.array):
    indexes = np.arange(1, x.size + 1, 1)
    return 1 + (1 / 4000) * np.sum(x ** 2) - np.prod(np.cos(x / np.sqrt(indexes)))


def ackley_func(x: np.array):
    n = x.size
    return -20 * np.exp(-0.2 * np.sqrt((1 / n) * np.sum(x ** 2))) - np.exp(
        (1 / n) * np.sum(np.cos(2 * math.pi * x))) + 20 + math.e


def schwefel_func(x: np.array):
    return np.sum(x ** 2) + np.prod(np.abs(x))


def u(z):
    a = 10
    k = 100
    m = 4
    result = 0
    size = z.size
    for cnt in range(size):
        if z[cnt] > a:
            result = result + k * (z[cnt] - a) ** m
        elif z[cnt] < (-1) * a:
            result = result + k * ((-1) * z[cnt] * (-1) * a) ** m
        else:
            result = result + 0

    return result


def leeyao_func(x: np.array):
    n = x.size
    xi = x[0:n - 1]
    xi_plus_1 = x[1:n]
    sigma1 = np.sum(((xi - 1) ** 2) * (1 + 10 * (np.sin(math.pi * xi_plus_1)) ** 2))

    return (math.pi / n) * (10 * ((np.sin(math.pi * x[1])) ** 2) + sigma1 + (x[n - 1] - 1) ** 2) + u(x)


# Press the green button in the gutter to run the script.
class Particle:
    def __init__(self, dimensions: int, x_from: float, x_to: float, func, ac_func):
        self.ac_func = ac_func
        self.func = func
        self.position = np.random.uniform(low=x_from, high=x_to, size=dimensions)
        self.velocity = np.random.uniform(low=x_from, high=x_to, size=dimensions)
        self.best_position = np.copy(self.position)
        self.best_score = math.inf
        self.interia_weight = 0.7

    def step(self, g, iteration_ratio):
        acc1 = self.ac_func(ACCELERATION_MAX, ACCELERATION_MIN, iteration_ratio)
        acc2 = self.ac_func(ACCELERATION_MAX, ACCELERATION_MIN, 1 - iteration_ratio)
        self.velocity = self.velocity * self.interia_weight + acc1 * random.uniform(0, 1) * (
                self.best_position - self.position) \
                        + acc2 * random.uniform(0, 1) * (g - self.position)
        self.position += self.velocity
        score = self.func(self.position)

        if random.uniform(0, 1) < 0.6:
            self.position = self.position + self.velocity
        else:
            # DOBRZE!?!?!?
            self.position = self.best_position + (1 * random.uniform(0, 1))

        if score < self.best_score:
            self.best_score = score
            self.best_position = np.copy(self.position)
        return score, self.position


class Swarm:
    def __init__(self, particle_number: int, dimensions: int, x_from: float, x_to: float, func, ac_func,
                 subswarm_number: int):
        self.ac_func = ac_func
        assert particle_number > 0
        assert dimensions > 0
        assert particle_number / 2 > subswarm_number
        assert particle_number % subswarm_number == 0
        self.particles_number_per_subswarm = int(particle_number / subswarm_number)
        self.dimensions = dimensions
        self.subswarms = [SubSwarm(self.particles_number_per_subswarm, self.dimensions, x_from, x_to, func, ac_func) for
                          _ in range(subswarm_number)]
        self.best_position = None
        self.best_score = math.inf

    def step(self, iteration_ratio: float):
        best_pos = None
        for subswarm in self.subswarms:
            score, position = subswarm.step(iteration_ratio)
            if score < self.best_score:
                self.best_score = score
                best_pos = position

        self.best_position = best_pos if best_pos is not None else self.best_position
        self.update_elite_particles()
        return self.best_score, best_pos

    def update_elite_particles(self):
        elite_particles = []
        for subswarm in self.subswarms:
            elite_particles.append(subswarm.elite_particle.best_position)
        mean = np.average(elite_particles)
        for subswarm in self.subswarms:
            for particle in subswarm.particles:
                if particle.best_score == subswarm.elite_particle.best_score:
                    particle.position = mean * (1 + random.uniform(0, 1))
                    subswarm.best_position = subswarm.elite_particle.position
                    subswarm.elite_particle.position = particle.position


class SubSwarm:
    def __init__(self, particle_number: int, dimensions: int, x_from: float, x_to: float, func, ac_func):
        self.particles = [Particle(dimensions, x_from, x_to, func, ac_func) for _ in range(particle_number)]
        self.best_score = math.inf
        self.elite_particle = self.particles[0]
        self.best_position = np.copy(self.particles[0].best_position)

    def step(self, iteration_ratio):

        for particle in self.particles:
            if particle != self.elite_particle:
                score, position = particle.step(self.best_position, iteration_ratio)
                if score < self.best_score:
                    self.best_score = score
                    self.elite_particle = particle
        return self.best_score, self.best_position


class MSPSOAlgorithm:
    def __init__(self, swarm: Swarm, iterations: int):
        self.swarm = swarm
        self.iterations = iterations

    def run(self):
        for i in range(self.iterations):
            best_score, _ = self.swarm.step(i / self.iterations)
            # print(f'{i}: {best_score}')
        return best_score, self.iterations


if __name__ == '__main__':
    swarm = Swarm(100, 10, -10, 10, griewank_func, linear_interpolation, 5)
    mspso_algorithm = MSPSOAlgorithm(swarm, iterations=1000)
    result = mspso_algorithm.run()
    print(result)
