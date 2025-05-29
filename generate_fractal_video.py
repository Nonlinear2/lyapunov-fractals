from lyapunov_core import *
import numpy as np
from random import randint

def main():
    pattern = "xyzzyxx"
    fractal_computer = ComputeFractals(x_min=3, x_max=3.9, y_min=3, y_max=3.9, z_min=3.4, 
                                       z_max=3.8, size=500, color_resolution=1000, pattern=pattern)
    video = fractal_computer.create_fractal_video(300)
    
    video[0].save(pattern + "_" + str(randint(0, 1_000)) + ".gif", save_all=True, optimize=True, append_images=video[1:], duration=50, loop=0)

if __name__ == "__main__":
    main()