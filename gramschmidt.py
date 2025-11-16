import numpy as np

def gram_schmidt(A):
    A = A.astype(float)
    m, n = A.shape
    Q = np.zeros((m, n))

    for j in range(n):
        v = A[:, j]

        for i in range(j):
            proj = np.dot(Q[:, i], v) * Q[:, i]
            v = v - proj

        Q[:, j] = v / np.linalg.norm(v)

    return Q


# example
A = np.array([[1,1,0],
              [1,0,1],
              [0,1,1]])

Q = gram_schmidt(A)
print(Q)
