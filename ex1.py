import sys, numpy as np, matplotlib.pyplot as plt

def distance(a, b):
    return np.sum((a-b) ** 2) # we don't need square root because it's a monotonic function

def kmeans(points, k: int, init_centroids, log_file: str):
    with open(log_file, 'w') as log:
        prev_centroids = None
        centroids = init_centroids.copy()
        iters = 0
        # while not converged or 20 iterations
        while iters < 20 and not np.array_equal(centroids, prev_centroids):
            clusters = [list() for _ in range(k)]
            prev_centroids = centroids
            # assign each point to the closest centroid
            for p_index,p in enumerate(points):
                min_index = 0
                min_dist = distance(p, centroids[0, :])
                for i,c in enumerate(centroids): # iterating the centroids
                    dist = distance(p, c)
                    if dist < min_dist:
                        min_index = i
                        min_dist = dist
                clusters[min_index].append(p)
            # update each centroid to be the average of the points in its cluster
            centroids = np.array([np.sum(c, axis=0) / len(c) if len(c) != 0 else centroids[clusters.index(c), :] for c in clusters]).round(4)
            
            print(f"[iter {iters}]:{','.join([str(i) for i in centroids])}", file=log)
            iters += 1
        
        return centroids, clusters

if __name__ == '__main__':
    if len(sys.argv) < 4:
        print('Not enough arguments!')
        exit(0)

    # arguments
    image_fname, centroids_fname, out_fname = sys.argv[1:4]
    # load centroids
    init_centroids = np.loadtxt(centroids_fname).round(4)
    k = init_centroids.shape[0]
    # load image
    orig_pixels = plt.imread(image_fname)
    pixels = orig_pixels.astype(float) / 255
    pixels = pixels.reshape(-1, 3)

    kmeans(pixels, k, init_centroids, out_fname)