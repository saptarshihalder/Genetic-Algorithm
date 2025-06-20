import numpy as np
import cv2
from dataclasses import dataclass
from typing import List, Tuple, Optional
import random
import matplotlib.pyplot as plt

@dataclass
class SegmentationMetrics:
    """Basic metrics for segmentation quality."""
    accuracy: float
    sensitivity: float
    specificity: float
    iou: float
    dice: float

class ImprovedGeneticAlgorithm:
    """GA for multi-threshold image segmentation."""

    def __init__(self, image: np.ndarray, num_thresholds: int = 2,
                 population_size: int = 50, generations: int = 100,
                 crossover_rate: float = 0.8, mutation_rate: float = 0.05):
        self.image = image
        self.num_thresholds = num_thresholds
        self.population_size = population_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.hist, _ = np.histogram(image, bins=256, range=(0, 255))
        self.hist = self.hist / self.hist.sum()
        self.population: List[dict] = []
        self.best: Optional[dict] = None
        self.fitness_history: List[float] = []

    # ------------------------- GA OPERATORS -------------------------
    def initialize_population(self):
        self.population = []
        for _ in range(self.population_size):
            t = np.sort(np.random.randint(1, 255, self.num_thresholds))
            self.population.append({'thr': t.astype(float), 'fitness': 0.0})

    def between_class_variance(self, t: np.ndarray) -> float:
        boundaries = np.concatenate([[0], t, [255]])
        total_mean = (self.hist * np.arange(256)).sum()
        var = 0.0
        for i in range(len(boundaries)-1):
            mask = (np.arange(256) >= boundaries[i]) & (np.arange(256) < boundaries[i+1])
            p = self.hist[mask].sum()
            if p > 0:
                mu = (self.hist[mask] * np.arange(256)[mask]).sum() / p
                var += p * (mu - total_mean) ** 2
        return var

    def evaluate_individual(self, ind: dict) -> float:
        t = np.clip(np.sort(ind['thr']), 1, 254)
        var = self.between_class_variance(t)
        # penalty for very close thresholds
        penalty = 0
        for i in range(len(t)-1):
            if t[i+1] - t[i] < 5:
                penalty += 0.1
        ind['thr'] = t
        ind['fitness'] = var - penalty
        return ind['fitness']

    def evaluate_population(self):
        for ind in self.population:
            self.evaluate_individual(ind)

    def select_parent(self) -> dict:
        self.population.sort(key=lambda x: x['fitness'], reverse=True)
        elite = max(1, int(0.2 * self.population_size))
        if random.random() < 0.3:
            return self.population[random.randint(0, elite-1)].copy()
        # tournament
        tour = random.sample(self.population, 5)
        return max(tour, key=lambda x: x['fitness']).copy()

    def crossover(self, p1: dict, p2: dict) -> Tuple[dict, dict]:
        if random.random() > self.crossover_rate:
            return p1.copy(), p2.copy()
        c1 = {'thr': np.zeros(self.num_thresholds), 'fitness': 0.0}
        c2 = {'thr': np.zeros(self.num_thresholds), 'fitness': 0.0}
        alpha = 0.5
        for i in range(self.num_thresholds):
            lo = min(p1['thr'][i], p2['thr'][i])
            hi = max(p1['thr'][i], p2['thr'][i])
            ran = hi - lo
            lower = max(1, lo - alpha * ran)
            upper = min(254, hi + alpha * ran)
            c1['thr'][i] = random.uniform(lower, upper)
            c2['thr'][i] = random.uniform(lower, upper)
        c1['thr'] = np.sort(c1['thr'])
        c2['thr'] = np.sort(c2['thr'])
        return c1, c2

    def mutate(self, ind: dict, gen: int):
        strength = 1.0 - 0.7 * gen / self.generations
        for i in range(self.num_thresholds):
            if random.random() < self.mutation_rate:
                delta = np.random.normal(0, 10 * strength)
                ind['thr'][i] = np.clip(ind['thr'][i] + delta, 1, 254)
        ind['thr'] = np.sort(ind['thr'])

    # ------------------------- MAIN LOOP -------------------------
    def run(self) -> np.ndarray:
        self.initialize_population()
        self.evaluate_population()
        for g in range(self.generations):
            self.population.sort(key=lambda x: x['fitness'], reverse=True)
            if self.best is None or self.population[0]['fitness'] > self.best['fitness']:
                self.best = self.population[0].copy()
            self.fitness_history.append(self.best['fitness'])
            new_pop: List[dict] = [ind.copy() for ind in self.population[:max(1, int(0.1*self.population_size))]]
            while len(new_pop) < self.population_size:
                p1 = self.select_parent()
                p2 = self.select_parent()
                c1, c2 = self.crossover(p1, p2)
                self.mutate(c1, g)
                self.mutate(c2, g)
                new_pop.extend([c1, c2])
            self.population = new_pop[:self.population_size]
            self.evaluate_population()
        return self.best['thr']

    # ------------------------- UTILITIES -------------------------
    def segment(self, thresholds: np.ndarray) -> np.ndarray:
        seg = np.zeros_like(self.image, dtype=np.uint8)
        t = np.concatenate([[0], thresholds, [255]])
        for i in range(len(t)-1):
            mask = (self.image >= t[i]) & (self.image < t[i+1])
            seg[mask] = i
        return seg

    def evaluate_segmentation(self, seg: np.ndarray, gt: np.ndarray) -> SegmentationMetrics:
        seg_bin = seg.flatten() > 0
        gt_bin = gt.flatten() > 0
        tp = np.sum(seg_bin & gt_bin)
        tn = np.sum(~seg_bin & ~gt_bin)
        fp = np.sum(seg_bin & ~gt_bin)
        fn = np.sum(~seg_bin & gt_bin)
        acc = (tp + tn) / (tp + tn + fp + fn)
        sens = tp / (tp + fn) if tp + fn > 0 else 0
        spec = tn / (tn + fp) if tn + fp > 0 else 0
        iou = tp / (tp + fp + fn) if tp + fp + fn > 0 else 0
        dice = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn > 0 else 0
        return SegmentationMetrics(acc, sens, spec, iou, dice)

# ------------- Helper utilities for examples/tests -------------
def preprocess_image(path: str) -> np.ndarray:
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3,3), 0)
    clahe = cv2.createCLAHE(2.0, (8,8))
    return clahe.apply(gray)

def synthetic_image(size: Tuple[int, int] = (128, 128)) -> np.ndarray:
    base = np.random.normal(128, 20, size).astype(np.uint8)
    for _ in range(20):
        c = (np.random.randint(10, size[0]-10), np.random.randint(10, size[1]-10))
        r = np.random.randint(5, 10)
        i = np.random.randint(40, 80)
        cv2.circle(base, c, r, i, -1)
    base = cv2.GaussianBlur(base, (5,5), 1)
    return base

def simple_ground_truth(img: np.ndarray, thr: int = 100) -> np.ndarray:
    return (img < thr).astype(np.uint8)


if __name__ == "__main__":
    # Example usage when running this module directly
    import matplotlib.pyplot as plt

    img = synthetic_image((256, 256))
    gt = simple_ground_truth(img)
    ga = ImprovedGeneticAlgorithm(img, num_thresholds=2, population_size=30, generations=50)
    thr = ga.run()
    seg = ga.segment(thr)

    print("Optimal thresholds:", thr)
    metrics = ga.evaluate_segmentation(seg, gt)
    print(f"Accuracy: {metrics.accuracy:.3f}")

    plt.subplot(1, 3, 1)
    plt.imshow(img, cmap="gray")
    plt.title("Image")
    plt.axis("off")
    plt.subplot(1, 3, 2)
    plt.imshow(seg, cmap="tab20")
    plt.title("Segmentation")
    plt.axis("off")
    plt.subplot(1, 3, 3)
    plt.plot(ga.fitness_history)
    plt.title("Fitness")
    plt.tight_layout()
    plt.show()

