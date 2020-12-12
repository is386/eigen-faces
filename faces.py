from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np

K = 10


def load_data():
    people = fetch_lfw_people(min_faces_per_person=20, resize=0.7)
    mask = np.zeros(people.target.shape, dtype=np.bool)
    for target in np.unique(people.target):
        mask[np.where(people.target == target)[0][:50]] = 1
    people_x = people.data[mask] / 255
    people_y = people.target[mask]
    return people_x, people_y


def pca(X, k):
    # Compute covariance matrix
    sigma = X.T @ X / (X.shape[0] - 1)
    # Compute eigenvectors and eigenvalues
    W, V = np.linalg.eig(sigma)
    # Get indices of sorted eigenvalues
    i = np.argsort(W)[::-1]
    # Sort eigenvectors based on eigenvalues
    V = V[:, i]
    W = W[i]
    # Return k eigenvalues and eigenvectors
    return W[0:k], V[:, 0:k]


# Return the primary principle component and plot it
def get_pc1(train_x):
    print("Reconstructing PC1...")
    # Get 2 eigenvectors
    W, V = pca(train_x, 2)

    # Primary principle component
    pc1 = np.reshape(V[:, 0], (5655, 1))
    plt.title("Primary Principle Component")
    plt.imshow(pc1.reshape((87, 65)), cmap="gray")
    plt.savefig("pc1.png")
    return pc1


# Use primary principle component to reconstruct first face
def reconstruct_pc1(train_x, scaler):
    pc1 = get_pc1(train_x)
    print("Reconstructing first face using PC1...")
    train_z = train_x @ pc1
    x_hat = train_z @ pc1.T
    x_hat = scaler.inverse_transform(x_hat)
    plt.title("Reconstruction of First Person using PC1")
    plt.imshow(x_hat[0, :].reshape((87, 65)), cmap="gray")
    plt.savefig("reconstruct1.png")


# Reconstructs first person using all components
def reconstruct_full(train_x, scaler):
    print("Reconstructing first face...")
    # Get all eigenvectors
    W, V = pca(train_x, len(train_x.T))

    # Sum all the eigenvalues
    sum_all_W = np.sum(W)
    sum_W = 0
    threshold = 0.95
    k = 0
    # Sums eigenvalues until we get the k most significant eigenvectors
    for w in W:
        k += 1
        sum_W += w
        if sum_W / sum_all_W >= threshold:
            break
    print("Components for 95% Reconstruction:", k)
    # Most significant eigenvectors
    V = V[:, 0:k]
    # Get projection for train data
    train_z = train_x @ V

    # Reconstructs train_x[0, :]
    x_hat = train_z @ V.T
    x_hat = scaler.inverse_transform(x_hat)
    plt.title("Reconstruction of First Person")
    plt.imshow(x_hat[0, :].reshape((87, 65)), cmap="gray")
    plt.savefig("reconstruct2.png")


# Returns the eigenvectors, clusters, and cluster centers
def create_clusters(X):
    print("Creating clusters...")
    C = []
    # Reduce data to 100D using PCA
    W, V = pca(X, 100)
    # Get projection for train and test data
    Z = X @ V

    # Pick K random samples
    np.random.seed(1)
    i = np.random.choice(range(0, len(Z)), K)

    # Reference vectors
    A = Z[i, :]

    # Will be used for termination
    new_A = np.array(A, copy=True)
    prev_sum = 0

    for iters in range(10000):
        C = [[] for i in range(K)]

        # Assign samples to clusters based on distance
        for x in Z:
            d = np.sum((x - A) ** 2, axis=1) ** 0.5
            i = np.argmin(d)
            C[i].append(x)

        # Compute new reference vectors
        for i, c in enumerate(C):
            new_A[i] = np.mean(c, axis=0)

        # Termination criteria
        new_sum = np.sum((new_A - A) ** 2) ** 0.5
        if abs(new_sum - prev_sum) < 2 ** (-23):
            break
        prev_sum = new_sum
        A = np.array(new_A, copy=True)

    print("Images Per Cluster:")
    for i, c in enumerate(C):
        print("Cluster {}: {}".format(i, len(c)))

    return C, A, V


def reconstruct_cluster_centers(A, V, scaler):
    print("Reconstructing cluster centers....")
    # Reconstruct the cluster centers
    x_hat = A @ V.T
    x_hat = scaler.inverse_transform(x_hat)

    # Plot images of the cluster centers
    fig, axes = plt.subplots(2, 5, figsize=(15, 8))
    for x, i, ax in zip(x_hat, range(len(x_hat)), axes.ravel()):
        ax.imshow(x.reshape((87, 65)), cmap="gray")
        ax.set_title("Cluster Center {}".format(i))
    fig.savefig("centers.png")


def reconstruct_cluster_min_max(C, A, V, scaler):
    print("Reconstructing cluster min and max...")
    maxes = []
    mins = []

    # Finds the images closest to and furthest from the cluster center
    for i, a, c in zip(range(K), A, C):
        c = np.asarray(c)
        d = np.sum((c - a) ** 2, axis=1) ** 0.5
        mx, mn = np.argmax(d), np.argmin(d)
        maxes.append(c[mx])
        mins.append(c[mn])

    # Reconstructs the min and max images
    x_max = maxes @ V.T
    x_min = mins @ V.T
    x_max = scaler.inverse_transform(x_max)
    x_min = scaler.inverse_transform(x_min)

    # Plot images closest to center
    fig, axes = plt.subplots(2, 5, figsize=(30, 16))
    for x, i, ax in zip(x_min, range(len(x_min)), axes.ravel()):
        ax.imshow(x.reshape((87, 65)), cmap="gray")
        ax.set_title("Min of Cluster {}".format(i))
    fig.savefig("min.png")

    # Plot images furthest from center
    fig, axes = plt.subplots(2, 5, figsize=(30, 16))
    for x, i, ax in zip(x_max, range(len(x_max)), axes.ravel()):
        ax.imshow(x.reshape((87, 65)), cmap="gray")
        ax.set_title("Max of Cluster {}".format(i))
    fig.savefig("max.png")


def main():
    # Process Data
    people_x, people_y = load_data()
    train_x, test_x, train_y, test_y = train_test_split(
        people_x, people_y, stratify=people_y, random_state=0)
    scaler = StandardScaler().fit(train_x)
    train_x, test_x = scaler.transform(train_x), scaler.transform(test_x)

    # Reconstruct first face using PC1
    reconstruct_pc1(train_x, scaler)

    # Reconstruct first face using all the components
    reconstruct_full(train_x, scaler)

    # Process whole dataset
    scaler = StandardScaler().fit(people_x)
    X = scaler.transform(people_x)

    # Creates clusters for all the faces
    C, A, V = create_clusters(X)

    # Reconstructs the cluster centers
    reconstruct_cluster_centers(A, V, scaler)

    # Reconstructs the min and max of each cluster
    reconstruct_cluster_min_max(C, A, V, scaler)


if __name__ == "__main__":
    main()
