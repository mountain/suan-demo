import numpy as np
import cv2

nx, ny = (32, 32)
xx, yy = np.meshgrid(np.linspace(0, 1, nx), np.linspace(0, 1, ny))
z = np.sin(0.5 * np.pi * (xx * xx + yy * yy))

for ix in range(10000):
    cv2.imwrite('data/z%04d.png' % ix, z * 255)
    z = (z < 0.5) * (2 * z) + (z >= 0.5) * (2 - 2 * z) + np.random.rand(32, 32) * 0.01
    z = z * (z < 1.0) * (z > 0.0)