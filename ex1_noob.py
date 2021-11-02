import sys, os, numpy as np
from matplotlib import pyplot as plt
from numpy.lib.function_base import average, iterable

# calculate the distance between two points
def distance(a, b):
    size = len(a)
    sum = 0
    for i in range(size):
        sum += (a[i] - b[i]) ** 2
    return sum ** 0.5

# compare two lists (of lists) recursievly 
def compare(a,b):
    if a is not iterable or b is not iterable:
        return a == b
    if len(a) != len(b):
        return False
    for i in range(len(a)):
        if compare(a[i], b[i]):
            return False
    return True


def kmeans(points, k, init_centroids):
    centroids = init_centroids
    converged = False
    # until convergence
    while not converged: 
        prev_centroids = list(centroids)
        clusters = [list() for _ in range(k)]
        # assign each point to the closest centroid
        for p in points:
            min = distance(p, centroids[0])
            min_index = 0
            for z in centroids:
                if distance(p,z) < min:
                    min = distance(p,z)
                    min_index = centroids.index(z)
            clusters[min_index].append(p)
        # update each centroid to be the average of the points in its cluster
        for i in range(len(centroids)):
            centroids[i] = list(np.average(clusters[i], 0))
        # check if anything change (if not, it converged)
        converged = compare(prev_centroids, centroids)
    return centroids, clusters


if __name__ == '__main__':
    image_fname, centroids_fname, out_fname = sys.argv[1:4]
    z = [list(x) for x in list(np.loadtxt(centroids_fname))]
    print(z)

    points = [[0,0,0], [1,1,1], [100,100,100]]
    print(kmeans(points, 2, z))
