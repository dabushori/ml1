import sys
import numpy as np
import matplotlib.pyplot as plt
import random

def create_cents(k: int):
    # return np.loadtxt('cents1.txt').round(4)
    if k in [2,4,8,16]:
        cents_dict = {
            2: [[0, 0, 0], [1, 1, 1]],
            4: [[0,0,0], [1,1,1], [0,1,0.5], [1,0,0.5]],
            8: [[0,0,0],[0,0,1],[0,1,0],[0,1,1],[1,0,0],[1,0,1],[1,1,0],[1,1,1]],
            16: [[0,0,0],[0,0,1],[0,1,0],[0,1,1],[1,0,0],[1,0,1],[1,1,0],[1,1,1],
                [0,0,0.3333],[0,1,0.3333],[1,0,0.3333],[1,1,0.3333],
                [1,0,0.6667],[0,1,0.6667],[1,1,0.6667],[0,0,0.6667]]
        }
        return np.array(cents_dict[k])
    return [[random.random() for _ in range(3)] for _ in range(k)]


def distance(a, b):
    # we don't need square root because it's a monotonic function
    return np.sum((a-b) ** 2)


def kmeans_per_iters(points, k: int, init_centroids):
    res = dict()
    prev_centroids = None
    centroids = np.array(init_centroids).copy()
    iters = 0
    # while not converged or 20 iterations
    while iters < 20 and not np.array_equal(centroids, prev_centroids):
        clusters = [list() for _ in range(k)]
        prev_centroids = centroids
        # assign each point to the closest centroid
        for p_index, p in enumerate(points):
            min_index = 0
            min_dist = distance(p, centroids[0, :])
            for i, c in enumerate(centroids):  # iterating the centroids
                dist = distance(p, c)
                if dist < min_dist:
                    min_index = i
                    min_dist = dist
            clusters[min_index].append(p)
        # update each centroid to be the average of the points in its cluster
        centroids = np.array([np.sum(c, axis=0) / len(c) if len(c) !=
                             0 else centroids[clusters.index(c), :] for c in clusters]).round(4)
        res[iters] = (centroids, clusters)
        print(
            f"[iter {iters}]:{','.join([str(i) for i in centroids])}")
        iters += 1
        # if iters < 20:
        #     print(f"[iter {iters}]:{','.join([str(i) for i in centroids])}", file=log)

    return res

    # return {
    #     1: (np.array([[1, 2]]), np.array([[0, 0], [1, 1]])),
    #     2: (np.array([[0.5, 0.5]]), np.array([[0, 0], [1, 1]]))
    # }  # iter : (centoids, clusters)


# params as numpy arrays
def cost_func(centroids, points):
    res = 0
    for p in points:
        min_val = np.dot(centroids[0] - p, centroids[0] - p)
        for cent in centroids:
            d = np.dot(cent - p, cent - p)
            min_val = d if d < min_val else min_val
        res += min_val
    return res


def is_in(p, cl):
    for i in cl:
        if np.equal(p, i).all():
            return True
    return False

def get_new_image(pixels, fin_cents, fin_clusts):
    new_im = list()
    for p in pixels:
        for i, cl in enumerate(fin_clusts):
            if is_in(p,cl):
                new_im.append(fin_cents[i])
                break
    return new_im


# python graph.py k image
if len(sys.argv) < 3:
    print('not enough paramters')
    exit(0)
k, image = sys.argv[1:3]
k = int(k)
orig_pixels = plt.imread(image)
pixels = orig_pixels.astype(float) / 255
pixels = pixels.reshape(-1, 3)
graph_x = list()
graph_y = list()
d = kmeans_per_iters(pixels, k, create_cents(k))
for iter, kmeans_res in d.items():
    graph_x.append(iter)
    cents, clusts = kmeans_res
    graph_y.append(cost_func(cents, pixels))




fig = plt.figure()

sub00 = fig.add_subplot(1, 2, 1)
sub00.set_title(f'cost/loss function for k={k}')
# plt.plot(x_vals, y_vals)
plt.plot(graph_x, graph_y, 'ro')
plt.xlabel('iterations')
plt.ylabel('cost/loss function')

print('here')

fin_cents, fin_clusts = list(d.values())[-1]
new_image = np.array(get_new_image(pixels, fin_cents, fin_clusts))
# new_image = np.array(pixels)
new_image = new_image.reshape(128,128,3)

sub01 = fig.add_subplot(1, 2, 2)
sub01.imshow(new_image)
# plt.imshow(plt.imread('dog.jpeg'))
plt.savefig(f'k{k}.png', format='png')
plt.show()
