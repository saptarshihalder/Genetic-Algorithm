import numpy as np
import random
from dataclasses import dataclass
from typing import List, Tuple
import matplotlib.pyplot as plt

@dataclass
class Task:
    """Represents a real-time task."""
    id: int
    arrival_time: float
    computation_time: float
    deadline: float
    laxity: float = 0.0

    def __post_init__(self):
        self.laxity = self.deadline - self.arrival_time - self.computation_time

@dataclass
class Processor:
    """Simple processor model."""
    id: int
    current_time: float = 0.0
    assigned_tasks: List[int] | None = None

    def __post_init__(self):
        if self.assigned_tasks is None:
            self.assigned_tasks = []

class Chromosome:
    """Chromosome encoding schedule order and processor mapping."""
    def __init__(self, num_tasks: int, num_processors: int):
        self.num_tasks = num_tasks
        self.num_processors = num_processors
        # Scheduling order of tasks
        self.scheduling = np.random.permutation(num_tasks)
        # Processor assignment for each task index
        self.mapping = np.random.randint(0, num_processors, num_tasks)
        self.fitness = float("inf")
        self.dlms = 0
        self.art = 0
        self.atat = 0
        self.schedule_length = 0

    def copy(self) -> "Chromosome":
        c = Chromosome(self.num_tasks, self.num_processors)
        c.scheduling = self.scheduling.copy()
        c.mapping = self.mapping.copy()
        c.fitness = self.fitness
        c.dlms = self.dlms
        c.art = self.art
        c.atat = self.atat
        c.schedule_length = self.schedule_length
        return c

class OPBGA:
    """Optimized Performance Based Genetic Algorithm."""

    def __init__(
        self,
        tasks: List[Task],
        num_processors: int,
        population_size: int = 100,
        generations: int = 200,
        crossover_rate: float = 0.8,
        mutation_rate: float = 0.1,
    ):
        self.tasks = tasks
        self.num_processors = num_processors
        self.population_size = population_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.population: List[Chromosome] = []
        self.best_chromosome: Chromosome | None = None
        self.fitness_history: List[float] = []
        self.total_comp_time = sum(t.computation_time for t in tasks)

    # --------------------- GA OPERATORS ---------------------
    def initialize_population(self):
        self.population = [
            Chromosome(len(self.tasks), self.num_processors)
            for _ in range(self.population_size)
        ]

    def evaluate_chromosome(self, chromosome: Chromosome) -> float:
        processors = [Processor(i) for i in range(self.num_processors)]

        # group tasks per processor following scheduling order
        task_groups = [[] for _ in range(self.num_processors)]
        for idx in chromosome.scheduling:
            proc_id = chromosome.mapping[idx]
            task_groups[proc_id].append(idx)

        dlms = 0
        total_rt = 0.0
        total_tat = 0.0
        max_completion = 0.0

        for proc_id, t_indices in enumerate(task_groups):
            current = 0.0
            for t_idx in t_indices:
                task = self.tasks[t_idx]
                start = max(current, task.arrival_time)
                end = start + task.computation_time
                if end > task.deadline:
                    dlms += 1
                rt = start - task.arrival_time
                tat = end - task.arrival_time
                total_rt += rt
                total_tat += tat
                current = end
                max_completion = max(max_completion, end)

        n = len(self.tasks)
        art = total_rt / n
        atat = total_tat / n
        chromosome.dlms = dlms
        chromosome.art = art
        chromosome.atat = atat
        chromosome.schedule_length = max_completion

        if dlms == 0:
            # normalised, multi-objective term (same as before)
            norm_len = max_completion / self.total_comp_time
            norm_art = art / self.total_comp_time
            norm_atat = atat / self.total_comp_time
            fitness = 0.25 * (norm_len + norm_art + norm_atat)
        else:
            # infeasible schedule – apply huge fixed penalty
            fitness = 1_000_000 * dlms

        chromosome.fitness = fitness
        return fitness

    def evaluate_population(self):
        for c in self.population:
            self.evaluate_chromosome(c)

    def roulette_wheel_selection(self) -> Chromosome:
        # stochastic uniform selection (roulette wheel)
        fitnesses = [1.0 / (c.fitness + 1e-6) for c in self.population]
        total = sum(fitnesses)
        probs = [f / total for f in fitnesses]
        r = random.random()
        cum = 0.0
        for idx, p in enumerate(probs):
            cum += p
            if r <= cum:
                return self.population[idx].copy()
        return self.population[-1].copy()

    def sus_selection(self, k: int = 2) -> List[Chromosome]:
        """Select k parents with stochastic-universal sampling."""
        fitnesses = np.array([1.0 / (c.fitness + 1e-6) for c in self.population])
        probs = fitnesses / fitnesses.sum()
        cumulative = np.cumsum(probs)
        start = random.random() / k
        pointers = start + np.arange(k) / k
        idx = [np.searchsorted(cumulative, p) for p in pointers]
        return [self.population[i].copy() for i in idx]

    def crossover(self, p1: Chromosome, p2: Chromosome) -> Tuple[Chromosome, Chromosome]:
        if random.random() > self.crossover_rate:
            return p1.copy(), p2.copy()

        size = len(p1.scheduling)
        cp = random.randint(1, size - 1)

        def order_cross(a: Chromosome, b: Chromosome) -> np.ndarray:
            child = [-1] * size
            child[:cp] = a.scheduling[:cp]
            pos = cp
            for gene in b.scheduling:
                if gene not in child:
                    child[pos] = gene
                    pos += 1
            return np.array(child)

        c1 = p1.copy()
        c2 = p2.copy()
        c1.scheduling = order_cross(p1, p2)
        c2.scheduling = order_cross(p2, p1)

        c1.mapping = np.concatenate([p1.mapping[:cp], p2.mapping[cp:]])
        c2.mapping = np.concatenate([p2.mapping[:cp], p1.mapping[cp:]])
        return c1, c2

    def mutate(self, chromosome: Chromosome):
        # 2a. swap two tasks in the order half
        if random.random() < self.mutation_rate:
            i1, i2 = random.sample(range(len(chromosome.scheduling)), 2)
            chromosome.scheduling[i1], chromosome.scheduling[i2] = (
                chromosome.scheduling[i2],
                chromosome.scheduling[i1],
            )
        # 2b. flip a processor assignment
        if random.random() < self.mutation_rate:
            j = random.randrange(len(chromosome.mapping))
            chromosome.mapping[j] = random.randrange(self.num_processors)

    # --------------------- GA EXECUTION ---------------------
    def run(self) -> Chromosome:
        self.initialize_population()
        self.evaluate_population()

        for gen in range(self.generations):
            self.population.sort(key=lambda c: c.fitness)
            if self.best_chromosome is None or self.population[0].fitness < self.best_chromosome.fitness:
                self.best_chromosome = self.population[0].copy()
            self.fitness_history.append(self.best_chromosome.fitness)

            new_pop: List[Chromosome] = []
            elite = int(0.1 * self.population_size)
            new_pop.extend([c.copy() for c in self.population[:elite]])

            while len(new_pop) < self.population_size:
                parent1, parent2 = self.sus_selection(2)
                child1, child2 = self.crossover(parent1, parent2)
                self.mutate(child1)
                self.mutate(child2)
                new_pop.extend([child1, child2])

            self.population = new_pop[: self.population_size]
            self.evaluate_population()
            # diversity restart
            if np.std([c.fitness for c in self.population]) < 1e-4:
                num_new = int(0.1 * self.population_size)
                self.population[-num_new:] = [
                    Chromosome(len(self.tasks), self.num_processors)
                    for _ in range(num_new)
                ]

            if gen % 10 == 0:
                print(
                    f"Generation {gen}: Best fitness = {self.best_chromosome.fitness:.4f}, "
                    f"DLMs = {self.best_chromosome.dlms}, "
                    f"Schedule Length = {self.best_chromosome.schedule_length:.2f}"
                )

        return self.best_chromosome

    # --------------------- VISUALIZATION ---------------------
    def visualize_schedule(self, chromosome: Chromosome):
        fig, ax = plt.subplots(figsize=(12, 6))
        task_groups = [[] for _ in range(self.num_processors)]
        for idx in chromosome.scheduling:
            task_groups[chromosome.mapping[idx]].append(idx)
        colors = plt.cm.Set3(np.linspace(0, 1, len(self.tasks)))
        for proc_id, idxs in enumerate(task_groups):
            current = 0.0
            for t_idx in idxs:
                task = self.tasks[t_idx]
                start = max(current, task.arrival_time)
                end = start + task.computation_time
                ax.barh(proc_id, task.computation_time, left=start, height=0.8,
                        color=colors[t_idx], edgecolor="black", linewidth=1)
                ax.text(start + task.computation_time/2, proc_id, f"T{t_idx}",
                        ha="center", va="center", fontsize=8)
                current = end
        ax.set_xlabel("Time")
        ax.set_ylabel("Processor")
        ax.set_yticks(range(self.num_processors))
        ax.set_yticklabels([f"P{i}" for i in range(self.num_processors)])
        ax.set_title("Task Schedule")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def plot_convergence(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.fitness_history)
        plt.xlabel("Generation")
        plt.ylabel("Best Fitness")
        plt.title("OPBGA Convergence")
        plt.grid(True, alpha=0.3)
        plt.show()

# --------------------- Example Usage ---------------------
def generate_random_tasks(num_tasks: int, max_comp: float = 20, max_arrival: float = 50) -> List[Task]:
    tasks = []
    for i in range(num_tasks):
        arrival = random.uniform(0, max_arrival)
        comp = random.uniform(1, max_comp)
        deadline = arrival + comp + random.uniform(5, 30)
        tasks.append(Task(i, arrival, comp, deadline))
    return tasks

if __name__ == "__main__":
    tasks = generate_random_tasks(20)
    ga = OPBGA(tasks, num_processors=3, population_size=50, generations=100)
    best = ga.run()
    print("\nBest Schedule Found:")
    print(f"Fitness: {best.fitness:.4f}")
    print(f"Deadline Misses: {best.dlms}")
    print(f"Average Response Time: {best.art:.2f}")
    print(f"Average Turnaround Time: {best.atat:.2f}")
    print(f"Schedule Length: {best.schedule_length:.2f}")
    ga.visualize_schedule(best)
    ga.plot_convergence()
