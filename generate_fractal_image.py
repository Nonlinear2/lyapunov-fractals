import numpy as np
from lyapunov_core import *
from random import randint

def main():
    fractal_computer = ComputeFractals(size=2000)
    image = fractal_computer.compute_fractal(1)

    image = Image.fromarray(np.swapaxes(image.astype(np.uint8), 0, 1))
    
    image.show()
    # image.save(pattern + '_' + str(randint(0, 1_000)) + '.png')

main()