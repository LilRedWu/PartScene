import numpy as np
from sklearn.cluster import DBSCAN

def generate_superpoints(point_cloud, eps=0.1, min_samples=10):
    coords = point_cloud[:, :3]  # assuming point_cloud is Nx6 (xyzrgb)
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(coords)
    labels = clustering.labels_
    
    num_superpoints = labels.max() + 1
    superpoints = np.zeros((len(point_cloud), num_superpoints), dtype=int)
    for i, label in enumerate(labels):
        if label != -1:  # ignore noise points
            superpoints[i, label] = 1
    
    return superpoints