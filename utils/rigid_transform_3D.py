#!/usr/bin/python
# pytorch version

# reference: https://github.com/nghiaho12/rigid_transform_3D/tree/master

import numpy as np
import torch

# Input: expects 3xN matrix of points
# Returns R,t
# R = 3x3 rotation matrix
# t = 3x1 column vector

def rigid_transform_3D(A, B):
    assert A.size() == B.size()

    num_rows, num_cols = A.size()
    if num_rows != 3:
        raise Exception(f"matrix A is not 3xN, it is {num_rows}x{num_cols}")

    num_rows, num_cols = B.size()
    if num_rows != 3:
        raise Exception(f"matrix B is not 3xN, it is {num_rows}x{num_cols}")

    # find mean column wise
    centroid_A = torch.mean(A, 1)
    centroid_B = torch.mean(B, 1)

    # ensure centroids are 3x1
    centroid_A = centroid_A.view(-1, 1)
    centroid_B = centroid_B.view(-1, 1)

    # subtract mean
    Am = A - centroid_A
    Bm = B - centroid_B

    H = torch.matmul(Am, torch.t(Bm))

    # sanity check
    #if linalg.matrix_rank(H) < 3:
    #    raise ValueError("rank of H = {}, expecting 3".format(linalg.matrix_rank(H)))

    # find rotation
    U, S, Vt = torch.linalg.svd(H)
    R = torch.matmul(torch.t(Vt), torch.t(U))

    # special reflection case
    if torch.linalg.det(R) < 0:
        # print("det(R) < R, reflection detected!, correcting for it ...")
        Vt[2,:] *= -1
        R = torch.matmul(torch.t(Vt), torch.t(U))

    t = -torch.matmul(R, centroid_A) + centroid_B

    return R, t