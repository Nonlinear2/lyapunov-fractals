# Lyapunov fractals
This is a simple python project to render lyapunov fractals. It relies on numba cuda kernels for relatively fast fractal generation.

![Alt text](./outputs/lyapunov_yyxxyyyyyzz.png?raw=true)
![Alt text](./outputs/lyapunov_xyyyxyxyy.png?raw=true)

## Requirements

- the [numba package with cuda](https://numba.pydata.org/numba-doc/latest/cuda/overview.html#requirements).
- numpy, pygame and pillow, that you can install with `pip install numpy pygame pillow`

and that's it!

## Features
### Image generation
Running the file `generate_fractal_image.py` creates an image of a fractal and displays it to the screen.
### Video generation
Running the file `generate_fractal_video.py` creates and stores a gif cycling through slices of a 3D lyapunov fractal in the z direction.
### Interactive pygame window
Running the file `fractal_zoom.py` opens a pygame window with a fractal. Press the left mouse button to zoom, the right mouse button to dezoom and tab to cycle through the z values of the fractal. To show the current coordinates, press space.

# Details
The main class is `ComputeFractals`. Its parameters are the following:
- x_min, x_max, y_min, y_max, z_min, z_max define the region in which fractals will be computed. These values need to be between 0 and 4.
- size: the size of the generated images in pixels.
- color_resolution: how many different shades of the color list `self.COLORS` are used.
- pattern: a string of "x", "y", and "z". The pattern defines which fractal is generated.
- num_iter: defines at which precision the pixel values are computed.

These parameters can be modified after the class construction by accessing them directly, except for:
- num_iter which can not be changed.
- the x and y boundaries, that you can change with the `set_region` method.
- the pattern, that you can change with the `set_pattern` method.

The class main method is `compute_fractal(z)`, which computes
a slice of a 3D lyapunov fractal at a given z value. This parameter doesn't matter if z doesn't appear in the pattern.

The class `FractalZoom` with the `run` method creates a pygame window and handles the zoom logic.