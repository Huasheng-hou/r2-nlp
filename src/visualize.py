import torch
import math
import numpy as np
from sklearn import manifold
import matplotlib.pyplot as plt


def TSNE(V, C, Y, cls=2):

    # 计算label embeddings之间的余弦相似度
    for idx in range(C.size()[0]):
        for j in range(idx + 1):
            s = torch.cosine_similarity(C[idx, :], C[j, :], dim=0)
            print("Cosine similarity between label {} and label {} is {:.6f}, angle is {:.4f}".format(
                idx, j, s, math.acos(s) / math.pi * 180))

    tsne = manifold.TSNE(n_components=2, init='pca', random_state=501)
    # fit = tsne.fit_transform(np.concatenate((V.detach().numpy(), C.detach().numpy())))
    # x_len = V.size()[0]
    # V_tsne = fit[0:x_len, :]
    # C_tsne = fit[x_len:, :]

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
        plt.scatter(C_tsne[idx:idx+1, 0], C_tsne[idx:idx+1, 1], marker='x', c='k', s=50)

    plt.show()


# V = np.load('../tmp/V.npy')
# C = np.load('../tmp/C.npy')
#
# tsne = manifold.TSNE(n_components=2, init='pca', random_state=501)
# V_tsne = tsne.fit_transform(V)
# C_tsne = tsne.fit_transform(C)
