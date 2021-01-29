import torch
import numpy as np
from sklearn import manifold
import matplotlib.pyplot as plt


def TSNE(V, C, Y, cls=2):
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=501)
    V_tsne = tsne.fit_transform(V.detach().numpy())
    C_tsne = tsne.fit_transform(C.detach().numpy())

    Z, colors = [], ['r', 'b', 'g', 'm', 'y', 'c', 'k']
    for idx in range(cls):
        Z.append([])

    for idx in range(len(Y)):
        y_h = Y[idx]
        Z[y_h].append(V_tsne[idx])

    for idx in range(cls):
        x = np.asarray(Z[idx])
        plt.scatter(x[:, 0], x[:, 1], marker='o', c=colors[idx], s=8)

    plt.show()


# V = np.load('../tmp/V.npy')
# C = np.load('../tmp/C.npy')
#
# tsne = manifold.TSNE(n_components=2, init='pca', random_state=501)
# V_tsne = tsne.fit_transform(V)
# C_tsne = tsne.fit_transform(C)
