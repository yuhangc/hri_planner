#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt


def generate_protocol(pt1, pt2, hpt1, hpt2, dd, n, save_path):
    direction = pt1 - pt2
    direction /= np.linalg.norm(direction)
    direction[2] = 0

    n /= 2

    # generate the points
    pt1_set = [pt1 + float(i) * dd * direction for i in range(-n, n+1)]
    pt2_set = [pt2 + float(i) * dd * direction for i in range(-n, n+1)]

    # attach random zeros/ones as intent
    for i in range(len(pt1_set)):
        k = np.random.randint(0, 2)
        if k == 0:
            pt1_set[i] = np.append(pt1_set[i], [0])
            pt2_set[i] = np.append(pt2_set[i], [1])
        else:
            pt1_set[i] = np.append(pt1_set[i], [1])
            pt2_set[i] = np.append(pt2_set[i], [0])

    pt1_set = np.asarray(pt1_set)
    pt2_set = np.asarray(pt2_set)

    # permute the points
    pt1_set = np.random.permutation(pt1_set)
    pt2_set = np.random.permutation(pt2_set)

    # plot to make sure
    plt.plot(pt1_set[:, 0], pt1_set[:, 1], "bo")
    plt.plot(pt2_set[:, 0], pt2_set[:, 1], "ro")
    hpt = np.vstack((hpt1, hpt2))
    plt.plot(hpt[:, 0], hpt[:, 1], "ko-")
    plt.axis("equal")
    plt.show()

    # generate the final output
    proto = [np.hstack(([0], pt2_set[-1, :3], hpt1, hpt2, 0))]
    for i in range(len(pt1_set)):
        proto.append(np.hstack(([i*2+1], pt1_set[i, :3], hpt1, hpt2, [pt1_set[i, 3]])))
        proto.append(np.hstack(([i*2+2], pt2_set[i, :3], hpt2, hpt1, [pt2_set[i, 3]])))

    proto = np.asarray(proto)

    # save to file
    np.savetxt(save_path, proto, delimiter=', ', fmt="%.3f")


if __name__ == "__main__":
    pt1 = np.array([0.160, 1.290, 2.490])
    pt2 = np.array([-4.200, 1.300, -0.390])
    hpt1 = np.array([-4.36, 4.61])
    hpt2 = np.array([0.22, -2.24])

    generate_protocol(pt1, pt2, hpt1, hpt2, 0.3, 5, "../../resources/exp_protocols/protocol-45.txt")
