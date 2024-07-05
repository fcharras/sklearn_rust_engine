import time
if __name__ == "__main__":
    from scipy.spatial.distance import cdist
    import numpy as np
    from sklearn_rust_engine import lloyd_iter_chunked_dense
    X = np.random.randn(30, 10)
    sample_weight = np.random.randn(30)
    centers_old = np.random.randn(15, 10)
    centers_new = np.random.randn(15, 10)
    weight_in_clusters = np.random.randn(15)
    labels = np.random.randint(0,15, size=(30,))
    centers_shift = np.random.randn(15,)
    n_threads = 4
    update_centers = True
    print(labels)

    tic = time.perf_counter()
    lloyd_iter_chunked_dense(X, centers_old, centers_new, labels, centers_shift, update_centers)
    print(time.perf_counter() - tic)
    print(labels)
    tic = time.perf_counter()
    distances = cdist(X, centers_old)
    closest = distances.argmin(axis=1)

    print(time.perf_counter() - tic)
    print(closest)
