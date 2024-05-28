import numpy as np
from lyapunov_core import *
from random import randint

def main():
    fractal_computer = ComputeFractals(pattern="yxxzxx", x_min = 3.43, x_max = 3.95, 
                                       y_min = 2.84, y_max = 3.36, size=2000, color_resolution=1900)
    image = fractal_computer.compute_fractal(1.57)

    image = Image.fromarray(np.swapaxes(image.astype(np.uint8), 0, 1))
    
    image.show()
    # image.save(pattern + '_' + str(randint(0, 1_000)) + '.png')

main()