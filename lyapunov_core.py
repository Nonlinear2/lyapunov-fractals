import numpy as np
from collections import Counter
from math import log
from numba import cuda
from PIL import Image
from bisect import bisect_left
from utils import ColorPalettes

class ComputeFractals:
    # @param x_min, x_max, y_min, y_max, z_min, z_max define the region in which
    # the fractal is computed. These values need to be between 0 and 4.
    # @param size the size of the image in pixels.
    # @param colors a list of hex colors
    # @param color_resolution how many different shades of self.COLORS are used
    # @param pattern a string of x, y, and z. the pattern defines which fractal is generated.
    # @num_iter at which precision are the pixel values computed.
    def __init__(self,
                 x_min=0.01,
                 x_max=4,
                 y_min=0.01,
                 y_max=4,
                 z_min=0.01,
                 z_max=4,
                 size = 500,
                 colors = ColorPalettes.red_orange_yellow,
                 color_resolution = 500,
                 pattern = "xxxyxxyy",
                 num_iter = 200,
                 verbose = False,
                 ):

        assert all([(v >= 0) and (v <= 4) for v in [x_min, x_max, y_min, y_max, z_min, z_max]])

        self.COLORS = colors

        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.z_min = z_min
        self.z_max = z_max

        self.size = size

        self.color_resolution = color_resolution

        self.num_iter = num_iter

        self.verbose = verbose

        self.set_pattern(pattern)

        y_space = np.tile(np.linspace(self.y_min, self.y_max, self.size), self.size).astype(np.float64)
        self.dev_y_space = cuda.to_device(y_space)

        x_space = np.repeat(np.linspace(self.x_min, self.x_max, self.size), self.size).astype(np.float64)
        self.dev_x_space = cuda.to_device(x_space)

        self.dev_output = cuda.device_array_like(self.dev_x_space)

        self.output = np.zeros_like(x_space)

        self.gpu = cuda.get_current_device()

        if (self.verbose):
            print(f"used GPU: {self.gpu.name.decode("utf-8")}")

    def get_color_idx(self, normalised_graph):
        split = np.linspace(0, 1, len(self.COLORS)+1)[:-1]
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

    def generate_gradient(self, frequence_map):
        switch_idx = self.get_color_idx(frequence_map/sum(frequence_map))

        gradient = []
        colors = iter(self.COLORS)
        for idx in switch_idx[1:]:
            col = self.hex_to_RGB(next(colors))
            gradient += [col]*(idx-len(gradient))
        gradient += [gradient[-1]]
        
        r, g, b = zip(*gradient)

        def smooth(y, box_pts):
            box = np.ones(box_pts)/box_pts
            y_smooth = np.convolve(y, box, mode='same')
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
        return gradient


    # @param pattern: a string of "x", "y" and "z"
    # for instance "xyxxyzzy"
    # @param num_iter: precision at which colors are computed at
    # each pixel. Increasing this number may reduce blurriness
    def set_pattern(self, pattern: str):
        assert set(list(pattern)).issubset({"x", "y", "z"})

        self.pattern = pattern

        sequence = tuple([{"x":0, "y":1, "z":2}[l] for l in pattern])
        len_sequence = len(sequence)
        num_iter = self.num_iter
        epsilon = 0.00001
        @cuda.jit
        def fractal_kernel(x_space, y_space, z):
            pos = cuda.grid(1)
            lambda_ = 0
            x_n = 0.5
            x = x_space[pos]
            y = y_space[pos]
            z = z[0]

            # in the following cases, the log is undefined
            # so we slightly modify the values
            if (abs(x - 0) < epsilon) or (abs(x - 2) < epsilon):
                x += epsilon
            if (abs(y - 0) < epsilon) or (abs(y - 2) < epsilon):
                y += epsilon
            if (abs(z - 0) < epsilon) or (abs(z - 2) < epsilon):
                z += epsilon
            
            for i in range(num_iter):
                r = (x, y, z)[sequence[i%len_sequence]]
                x_n = r*x_n*(1-x_n)
                lambda_ += log(abs(r*(1-2*x_n)))
            x_space[pos] = lambda_  
        
        self.fractal_kernel = fractal_kernel
        
    def set_region(self, x_min, x_max, y_min, y_max):

        assert all([(v >= 0) and (v <= 4) for v in [x_min, x_max, y_min, y_max]])

        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max

        y_space = np.tile(np.linspace(self.y_min, self.y_max, self.size), self.size).astype(np.float64)
        self.dev_y_space = cuda.to_device(y_space)

        x_space = np.repeat(np.linspace(self.x_min, self.x_max, self.size), self.size).astype(np.float64)
        self.dev_x_space = cuda.to_device(x_space)
    
    def get_gradient(self, indexes):
        np.random.seed(21)
        lambda_count = dict(Counter(np.random.choice(indexes, min(indexes.size, 100_000)))) # sample only 100_000 values of image to make
        # the gradient. this improves performance
        frequence = np.array([lambda_count.get(i, 0) for i in range(self.color_resolution)])
        gradient = self.generate_gradient(frequence)
        return gradient

    def compute_fractal(self, z):

        assert (0 <= z) and (z <= 4)

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
            print('grid stride loops not implemented')
            exit()

        self.dev_output.copy_to_device(self.dev_x_space)
        dev_z = cuda.to_device(np.array([z]).astype(np.float64))

        if (self.verbose):
            print("copied data to GPU, executing fractal kernel")

        self.fractal_kernel[blockspergrid, threadsperblock](self.dev_output, self.dev_y_space, dev_z)
        cuda.synchronize()

        if (self.verbose):
            print("fractal computed, copying data back to cpu")

        self.dev_output.copy_to_host(self.output)

        if (self.verbose):
            print("data copied, computing color gradient")

        lambda_min = np.amin(self.output)
        scaling_factor = np.amax(self.output) - lambda_min
        if (scaling_factor == 0):
            return np.zeros((self.size, self.size, 3))
        indexes = ((self.color_resolution-1)*(self.output-lambda_min) / scaling_factor).astype(int)
        gradient = self.get_gradient(indexes)
        image = gradient[indexes].reshape((self.size, self.size, 3))
        image = np.flip(image, axis=1)

        return image
    
    def create_fractal_video(self, num_frames):
        video = []
        for idx, z in enumerate(np.linspace(self.z_min, self.z_max, num_frames), 1):
            image = Image.fromarray(self.compute_fractal(z).astype(np.uint8)).convert('RGB')
            video.append(image)
            if self.verbose:
                print(f"frame {idx}/{num_frames}", end="\r")
        if self.verbose:
            print()
        return video