import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn as sk

original_image = plt.imread('spidey.jpg')
print("size of img: ", original_image.shape)
X_resource = np.reshape(original_image, (original_image.shape[0] * original_image.shape[1], original_image.shape[2]))

def random_initialize(X,K):
    m,n = X.shape
    random = np.random.choice(m,K,replace=False)
    centroids = X[random]
    return centroids

def find_closest_centroids(X,centroids):
    distances = np.sum((X[:, np.newaxis, :] - centroids) ** 2, axis=2)
    return np.argmin(distances, axis=1)

def redefine_centroids(X,K,idx):
    m,n = X.shape
    centroids = np.zeros((K,n))
    for k in range(K):
        new_centroid = np.mean(X[idx == k],axis = 0)
        centroids[k] = new_centroid
    return centroids

def run_kMeans(X,K,interators):
    centroids = random_initialize(X,K)
    idx = np.zeros(X.shape[0])
    for i in range(interators):
        idx = find_closest_centroids(X, centroids)
        centroids = redefine_centroids(X,K,idx)

    return centroids,idx

K = 5
max_iters = 10
centroids, idx = run_kMeans(X_resource, K, max_iters)

X_recovered = centroids[idx,:]
X_recovered = np.reshape(X_recovered, original_image.shape)
fig, ax = plt.subplots(1,2, figsize=(16,16))
plt.axis('off')

ax[0].imshow(original_image)
ax[0].set_title('Original')
ax[0].set_axis_off()

ax[1].imshow(X_recovered / 255.0)
ax[1].set_title('Compressed with %d colours'%K)
ax[1].set_axis_off()

plt.show()