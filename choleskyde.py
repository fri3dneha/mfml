import numpy as np

def cholesky(A):
    A = A.astype(float)
    n = len(A)
    L = np.zeros((n, n))

    for i in range(n):
        for j in range(i+1):
            s = 0
            for k in range(j):
                s += L[i][k] * L[j][k]

            if i == j:
                L[i][j] = (A[i][i] - s) ** 0.5
            else:
                L[i][j] = (A[i][j] - s) / L[j][j]

    return L


# example
A = np.array([[4,12,-16],
              [12,37,-43],
              [-16,-43,98]])

L = cholesky(A)
print(L)
