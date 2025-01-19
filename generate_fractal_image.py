import numpy as np
from lyapunov_core import *

def main():
    pattern = "yyxxyyyyyzz"
    x_min, x_max, y_min, y_max = 1.009, 1.244, 3.662, 3.898
    z = 2.86

    fractal_computer = ComputeFractals(pattern=pattern, 
        x_min = x_min, x_max = x_max, y_min = y_min, y_max = y_max,
        size=8000, color_resolution=1900, num_iter=600
    )

    image = fractal_computer.compute_fractal(z, verbose=True)

    image = Image.fromarray(np.swapaxes(image.astype(np.uint8), 0, 1))
    
    image.show()

    if (input() == "y"):
        image.save(str(pattern) + '_' + 
                str(round(x_min, 3)) + '_' + 
                str(round(x_max, 3)) + '_' + 
                str(round(y_min, 3)) + '_' + 
                str(round(y_max, 3)) + '_z_' +
                str(round(z, 3)) + "_" + 
                "-".join(fractal_computer.COLORS) + '.png')
        print("image saved")

main()