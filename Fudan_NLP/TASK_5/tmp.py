import torch
import numpy as np

from torchtext.vocab import Vectors

def draw_pict(y_data):
    from matplotlib import pyplot as plt
    x = np.arange(0, len(y_data), 1, dtype=np.int32)
    y = np.array(y_data)

    plt.plot(x, y, linestyle='-')
    plt.show()

y_data = [2, 4, 6, 8, 10]
draw_pict(y_data)