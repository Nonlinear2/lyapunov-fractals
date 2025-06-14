import numpy as np
from collections import Counter
from math import log
from numba import cuda
from PIL import Image
from bisect import bisect_left
from utils import ColorPalettes

class ComputeFractals:
    DEFAULT_PARAMS = {
        "pattern": "xxxyxxyy",
        "x_min": 0.01,
        "x_max": 4,
        "y_min": 0.01,
        "y_max": 4,
        "z": 2.81,
        "size": 500,
        "color_resolution": 500,
        "num_iter": 200,
        "colors": ColorPalettes.red_orange_yellow
    }

    def __init__(self):

        self.gpu = cuda.get_current_device()

        self.progress_callback = None
        self.total_stage_number = 4

        self.set_parameters(**ComputeFractals.DEFAULT_PARAMS)

    # @param x_min, x_max, y_min, y_max, z define the region in which
    # the fractal is computed. These values need to be between 0 and 4.
    # @param size the size of the image in pixels.
    # @param colors a list of hex colors such as "#ff0ed6"
    # @param color_resolution how many different shades of self.colors are used
    # @param pattern a string of x, y, and z. the pattern defines which fractal is generated.
    # @num_iter at which precision are the pixel values computed.
    def set_parameters(self, **kwargs):
        for key in kwargs:
            if key not in {"x_min", "x_max", "y_min", "y_max", "z", "size", "colors",
                        "color_resolution", "pattern", "num_iter"}:
                raise TypeError(f"set_parameters() got an unexpected argument '{key}'")

        def is_new(attr):
           return attr in kwargs and kwargs[attr] != getattr(self, attr, None)

        if is_new("z"):
            self.z = kwargs["z"]
            self.dev_z = cuda.to_device(np.array([self.z]).astype(np.float64))
    
        if is_new("pattern") or is_new("num_iter"):
            for param in {"pattern", "num_iter"}:
                if param in kwargs:
                    setattr(self, param, kwargs[param])
            self.recompute_fractal_kernel()

        new_size = is_new("size")

        # update y space if needed
        if is_new("y_min") or is_new("y_max") or new_size:
            for param in {"y_min", "y_max", "size"}:
                if param in kwargs:
                    setattr(self, param, kwargs[param])
            y_space = np.tile(np.linspace(self.y_min, self.y_max, self.size), self.size).astype(np.float64)
            self.dev_y_space = cuda.to_device(y_space)

        # update x space if needed
        if is_new("x_min") or is_new("x_max") or new_size:
            for param in {"x_min", "x_max", "size"}:
                if param in kwargs:
                    setattr(self, param, kwargs[param])
            x_space = np.repeat(np.linspace(self.x_min, self.x_max, self.size), self.size).astype(np.float64)
            self.dev_x_space = cuda.to_device(x_space)
        
        # update output if needed
        if new_size:
            self.size = kwargs["size"]
            # x space is arbitrary, only the shape matters
            self.dev_output = cuda.device_array_like(self.dev_x_space)
            self.output = np.zeros_like(x_space)

        # set remaining attributes 
        for param in {"color_resolution", "colors"}:
            if param in kwargs:
                setattr(self, param, kwargs[param])

        assert set(list(self.pattern)).issubset({"x", "y", "z"})
        assert all([(v > 0) and (v <= 4) for v in [self.x_min, self.x_max, self.y_min, self.y_max, self.z]])
        assert self.color_resolution > 1

    def set_progress_callback(self, progress_callback):
        self.progress_callback = progress_callback

    def get_color_idx(self, normalised_graph):
        split = np.linspace(0, 1, len(self.colors)+1)[:-1]
        integral_0_to_x = np.cumsum(normalised_graph)

        color_switch_idx = []
        curr_position = 0
        for threshold in split:
            current_position = bisect_left(integral_0_to_x, threshold, curr_position)
            color_switch_idx.append(current_position)

        color_switch_idx += [len(normalised_graph)-1] # add last element
        return color_switch_idx

    def hex_to_RGB(self, hex_str):
        #Pass 16 to the integer function for change of base
        return [int(hex_str[i:i+2], 16) for i in range(1,6,2)]

    # recompute needed when pattern or num_iter is changed
    def recompute_fractal_kernel(self):
        sequence = tuple([{"x":0, "y":1, "z":2}[l] for l in self.pattern])
        len_sequence = len(sequence)
        num_iter = self.num_iter
        epsilon = 0.00001
        @cuda.jit
        def fractal_kernel(x_space, y_space, z):
            pos = cuda.grid(1)
            lambda_N = 0
            x_n = 0.5
            x = x_space[pos]
            y = y_space[pos]
            z = z[0]

            # in the following cases, the log is undefined
            # so we slightly modify the values
            if abs(x - 2) < epsilon:
                x += epsilon
            if abs(y - 2) < epsilon:
                y += epsilon
            if abs(z - 2) < epsilon:
                z += epsilon
            
            for i in range(num_iter):
                r = (x, y, z)[sequence[i%len_sequence]]
                x_n = r*x_n*(1-x_n)
                lambda_N += log(abs(r*(1-2*x_n)))
            x_space[pos] = lambda_N
        
        self.fractal_kernel = fractal_kernel

    def compute_fractal(self):
        if self.size**2 <= self.gpu.MAX_THREADS_PER_BLOCK:
            blockspergrid = 1
            threadsperblock = self.size**2
        elif self.size <= self.gpu.MAX_THREADS_PER_BLOCK:
            blockspergrid = self.size
            threadsperblock = self.size
        elif self.size**2 <= self.gpu.MAX_GRID_DIM_X * self.gpu.MAX_THREADS_PER_BLOCK:
            threadsperblock = self.gpu.MAX_THREADS_PER_BLOCK
            blockspergrid = (self.size**2 + (threadsperblock-1)) // threadsperblock
        else:
            print("grid stride loops not implemented")
            exit()

        if self.progress_callback != None:
            self.progress_callback("Copying coordinates to the GPU", 1)
    
        self.dev_output.copy_to_device(self.dev_x_space)

        if self.progress_callback != None:
            self.progress_callback("Executing fractal kernel (may take a while)", 2)

        self.fractal_kernel[blockspergrid, threadsperblock](self.dev_output, self.dev_y_space, self.dev_z)
        cuda.synchronize()

        if self.progress_callback != None:
            self.progress_callback("Copying data back to the CPU", 3)

        self.dev_output.copy_to_host(self.output)

        if self.progress_callback != None:
            self.progress_callback("Applying color gradient", 4)

        lambda_min = np.amin(self.output)
        scaling_factor = np.amax(self.output) - lambda_min
        if (scaling_factor == 0):
            return np.zeros((self.size, self.size, 3))
        self.indexes = ((self.color_resolution-1)*(self.output-lambda_min) / scaling_factor).astype(int)

        np.random.seed(21)
        # improve performance by sampling only 50_000 values of image to make the gradient
        lambda_count = dict(Counter(np.random.choice(self.indexes, min(self.indexes.size, 50_000)))) 
        frequence_map = np.array([lambda_count.get(i, 0) for i in range(self.color_resolution)])

        self.switch_idx = self.get_color_idx(frequence_map/sum(frequence_map))


    def apply_gradient(self, colors = None):
        if colors is None:
            colors = self.colors

        gradient = []
        colors = iter(colors)
        for idx in self.switch_idx[1:]:
            col = self.hex_to_RGB(next(colors))
            gradient += [col]*(idx-len(gradient))
        gradient += [gradient[-1]]
        
        r, g, b = zip(*gradient)

        def smooth(y, box_pts):
            box = np.ones(box_pts)/box_pts
            y_smooth = np.convolve(y, box, mode="same")
            return y_smooth
        
        # this coefficient changes how smooth the color transitions are
        box_pts = 50
        for _ in range(3):
            r = smooth(r, box_pts)
            g = smooth(g, box_pts)
            b = smooth(b, box_pts)

        r = np.rint(r).astype(int)
        g = np.rint(g).astype(int)
        b = np.rint(b).astype(int)

        gradient = np.array(list(zip(r, g, b)))

        image = gradient[self.indexes].reshape((self.size, self.size, 3))
        image = np.flip(image, axis=1)

        return image.astype(np.uint8)


    def create_fractal_video(self, z_min, z_max, num_frames, verbose = True):
        assert all([(v > 0) and (v <= 4) for v in [z_min, z_max]])
        assert z_min < z_max

        video = []
        for idx, z in enumerate(np.linspace(z_min, z_max, num_frames), 1):
            self.set_parameters(z=z)
            self.compute_fractal()
            image = Image.fromarray(self.apply_gradient()).convert("RGB")
            video.append(image)
            if verbose:
                print(f"frame {idx}/{num_frames}", end="\r")
        if verbose:
            print()
        return video