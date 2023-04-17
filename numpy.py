import numpy as np
N = 200
x_clean = np.sin(np.arange(N)/20.)
x_noisy = x_clean + .05*np.random.randn(N)
w = 1./3*np.ones(3)
