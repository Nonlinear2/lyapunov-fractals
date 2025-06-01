# Lyapunov fractals
This is a simple python project to render lyapunov fractals. It relies on numba cuda kernels for relatively fast fractal generation.

![Alt text](./outputs/lyapunov_yyxxyyyyyzz.png?raw=true)
![Alt text](./outputs/lyapunov_xyyyxyxyy.png?raw=true)

## Requirements

As this project uses numba-cuda, you need a CUDA-enabled NVIDIA graphics card with Compute Capability 3.5 or greater.

## Installation
Optionally, you can create a virtual environement with
```
python -m venv .venv
.venv\Scripts\activate
```

Then run 
```pip install -r requirements.txt```
to install all the necessary packages.

## Features
### Interactive pyside window
Running the file `fractal_zoom.py` opens a window where you can modify the fractal parameters:
- Press the `left mouse button` to zoom
- Press the `right mouse button` to dezoom
- Press `space` to increase the $z$ coordinate of the fractal
- Press `backspace` to decrease the $z$ coordinate of the fractal
- Press `c` to cycle the fractal pattern.

### Image generation
Running the file `generate_fractal_image.py` creates an image of a fractal and displays it to the screen.
### Video generation
Running the file `generate_fractal_video.py` creates and stores a gif cycling through slices of a 3D lyapunov fractal in the $z$ direction.

# Details
## Theory
A 2D lyapunov fractal is created by computing the lyapunov exponent for each screen pixel of modified logistic sequences.
Given a pattern and an $x$, $y$ position on the screen, these sequences are defined as 
$$x_{n+1} = r_{x, y}(n) x_n (1 - x_n)$$
where $r_{x, y}(n) = pattern[n]$
Then, the pixel is colored according to the value of the lyapunov exponent.

More precisely, the lyapunov exponent is calculated using 
$$\lambda = \lim_{N \to +\infty} \frac{1}{N}\sum_{n=1}^{N}\ln|r_n\cdot(1-2x_n)|$$
[...]

Interestingly, the diagonal of a 2D lyapunov fractal is always the same, it is given by the lyapunov exponents of the logistic for different values of $r$.

## Implementation
You can set the fractal parameters of `ComputeFractals` using the `set_parameters` method, which can take in any combination of:
- `x_min`, `x_max`, `y_min`, `y_max`, `z` define the region in which fractals will be computed. These values need to be between 0 and 4.
- `size`: the size of the generated images in pixels.
- `colors`: a list of hex colors
- `color_resolution`: how many different shades of the color list are used.
- `pattern`: a string of "x", "y", and "z". The pattern defines which fractal is generated.
- `num_iter`: defines at which precision the pixel values are computed.

Calling `compute_fractal()` then computes a slice of a 3D lyapunov fractal using the current parameters.