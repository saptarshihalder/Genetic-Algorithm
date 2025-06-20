import numpy as np
from pathology_ga import ImprovedGeneticAlgorithm, synthetic_image, simple_ground_truth


def test_basic_run():
    img = synthetic_image((64, 64))
    gt = simple_ground_truth(img)
    ga = ImprovedGeneticAlgorithm(img, num_thresholds=2, population_size=10, generations=5)
    thr = ga.run()
    assert 2 <= len(thr) <= 6
    seg = ga.segment(thr)
    metrics = ga.evaluate_segmentation(seg, gt)
    assert 0 <= metrics.accuracy <= 1
