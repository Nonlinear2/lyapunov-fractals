from lyapunov_core import *

def main():
    fractal_zoom = FractalZoom(z_min = 2.2, pattern = "xxxyzzyx", size = 500, color_resolution=1800)
    fractal_zoom.run(0.01)

if __name__ == "__main__":
    main()