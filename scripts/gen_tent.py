import numpy as np
import cv2


for ix in range(800):
    z = np.random.rand(32, 32)
    for jx in range(12):
        z = (z < 0.5) * (2 * z) + (z >= 0.5) * (2 - 2 * z) + np.random.rand(32, 32) * 0.01
        z = z * (z < 1.0) * (z > 0.0)
        cv2.imwrite('data/tent/train/z%03d_%02d.png' % (ix, jx), z * 255)

for ix in range(200):
    z = np.random.rand(32, 32)
    for jx in range(12):
        z = (z < 0.5) * (2 * z) + (z >= 0.5) * (2 - 2 * z) + np.random.rand(32, 32) * 0.01
        z = z * (z < 1.0) * (z > 0.0)
        cv2.imwrite('data/tent/validation/z%03d_%02d.png' % (ix, jx), z * 255)

for ix in range(500):
    z = np.random.rand(32, 32)
    for jx in range(12):
        z = (z < 0.5) * (2 * z) + (z >= 0.5) * (2 - 2 * z) + np.random.rand(32, 32) * 0.01
        z = z * (z < 1.0) * (z > 0.0)
        cv2.imwrite('data/tent/test/z%03d_%02d.png' % (ix, jx), z * 255)
