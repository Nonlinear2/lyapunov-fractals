import numpy as np
from lyapunov_core import *

def main():
    pattern = "yyxxyyyyyzz"
    x_min, x_max, y_min, y_max = 1.009, 1.244, 3.662, 3.898

    fractal_computer = ComputeFractals(pattern=pattern, 
        x_min = x_min, x_max = x_max, y_min = y_min, y_max = y_max,
        size=8000, color_resolution=1900, num_iter=600
    )

    image = fractal_computer.compute_fractal(2.86)

    image = Image.fromarray(np.swapaxes(image.astype(np.uint8), 0, 1))
    
    # image.show()
    # image.save(pattern + '_' + 
    #            round(x_min, 3) + '_' + 
    #            round(x_max, 3) + '_' + 
    #            round(y_min, 3) + '_' + 
    #            round(y_max, 3) + '.png')

main()