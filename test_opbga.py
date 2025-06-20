import random
import numpy as np
from opbga import OPBGA, generate_random_tasks, Chromosome


def test_generate_random_tasks():
    random.seed(1)
    tasks = generate_random_tasks(3, max_comp=2, max_arrival=5)
    assert len(tasks) == 3
    assert {t.id for t in tasks} == {0, 1, 2}


def test_opbga_run_small():
    random.seed(0)
    np.random.seed(0)
    tasks = generate_random_tasks(5, max_comp=2, max_arrival=5)
    ga = OPBGA(tasks, num_processors=2, population_size=10, generations=5)
    best = ga.run()
    assert isinstance(best, Chromosome)
    assert best.schedule_length > 0
