import numpy as np
from lyapunov_core import *

def main():
    pattern = "xyxzyxzyxzyyzxyxzzyxyzxyxzyzxyz"
    x_min, x_max, y_min, y_max = 2.1472, 2.2895, 3.7194, 3.861744
    z = 2.86

    fractal_computer = ComputeFractals()

    fractal_computer.set_parameters(pattern=pattern, 
        x_min = x_min, x_max = x_max, y_min = y_min, y_max = y_max,
        size=1960, color_resolution=1900, num_iter=3000, colors = ColorPalettes.black_purple)

    fractal_computer.compute_fractal()

    image = fractal_computer.apply_gradient()

    image = Image.fromarray(np.swapaxes(image.astype(np.uint8), 0, 1))
    
    image.show()

    print("save the image? y/n")
    if (input() == "y"):
        image.save(str(pattern) + "_" + 
                str(round(x_min, 3)) + "_" + 
                str(round(x_max, 3)) + "_" + 
                str(round(y_min, 3)) + "_" + 
                str(round(y_max, 3)) + "_z_" +
                str(round(z, 3)) + "_" + 
                "-".join(fractal_computer.colors) + ".png")
        print("image saved")

main()