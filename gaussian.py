import numpy as np

def gaussian_elimination(a):
    a = a.astype(float)
    n = len(a)

    for i in range(n):
        # pivot
        if a[i][i] == 0:
            for k in range(i+1, n):
                if a[k][i] != 0:
                    a[[i, k]] = a[[k, i]]
                    break

        # eliminate below
        for j in range(i+1, n):
            ratio = a[j][i] / a[i][i]
            for k in range(i, a.shape[1]):
                a[j][k] = a[j][k] - ratio * a[i][k]

    return a


# example
aug = np.array([[2,1,-1,8],
                [-3,-1,2,-11],
                [-2,1,2,-3]])

ref = gaussian_elimination(aug)
print(ref)
