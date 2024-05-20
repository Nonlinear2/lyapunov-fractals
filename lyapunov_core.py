import numpy as np
import pygame
from collections import Counter
from math import log
from numba import cuda
from PIL import Image
from bisect import bisect_left

class ComputeFractals:
    # @param x_min, x_max, y_min, y_max, z_min, z_max define the region in which
    # the fractal is computed. These values need to be between 0 and 4.
    # @param size the size of the image in pixels.
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
                 color_resolution = 500,
                 pattern = "xxxyxxyy", 
                 num_iter = 200,
                 ):

        assert all([(v >= 0) and (v <= 4) for v in [x_min, x_max, y_min, y_max, z_min, z_max]])


        self.COLORS = ["#03071e", "#370617", "#6a040f", "#9d0208", "#d00000", "#FF7F50", "#B8860B", "#FFC700", "#bdb76b", "#6b8e23", "#556b2f", "#006600", "#004d00"]    
        #self.COLORS = ['#000000','#1A0028', '#2E0854', '#4B0082', '#6A0DAD', '#8A2BE2', '#9370DB', '#BA55D3', '#DA70D6', '#FFB6C1', '#F08080', '#FA8072', '#FF7F50', '#FFA500', '#FF9500', '#FFBF00', '#DAA520', '#FFD700']
        
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.z_min = z_min
        self.z_max = z_max

        self.size = size

        self.color_resolution = color_resolution

        self.num_iter = num_iter

        self.set_pattern(pattern)

        y_space = np.tile(np.linspace(self.y_min, self.y_max, self.size), self.size).astype(np.float64)
        self.dev_y_space = cuda.to_device(y_space)

        x_space = np.repeat(np.linspace(self.x_min, self.x_max, self.size), self.size).astype(np.float64)
        self.dev_x_space = cuda.to_device(x_space)

        self.dev_output = cuda.device_array_like(self.dev_x_space)

        self.output = np.zeros_like(x_space)

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
        @cuda.jit
        def fractal_kernel(x_space, y_space, z):
            pos = cuda.grid(1)
            lambda_ = 0
            x_n = 0.5
            x = x_space[pos]
            y = y_space[pos]
            z = z[0]
            if (x == 2) or (y == 2) or (z == 2) or (x == 0) or (y == 0) or (z == 0):
                # in this case the log is undefined
                x_space[pos] = 0 # color the pixel black
            else:
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
        lambda_count = dict(Counter(np.random.choice(indexes, indexes.size//100))) # sample only 1% of values of image to make
        # the gradient. this improves performance
        frequence = np.array([lambda_count.get(i, 0) for i in range(self.color_resolution)])
        gradient = self.generate_gradient(frequence)
        return gradient

    def compute_fractal(self, z):

        assert (0 <= z) and (z <= 4)

        if self.size**2 <= 256:
            blockspergrid = 1
            threadsperblock = self.size**2
        elif self.size <= 256:
            blockspergrid = self.size
            threadsperblock = self.size
        elif self.size**2 <= 32768*256:
            threadsperblock = 256
            blockspergrid = (self.size**2 + (threadsperblock-1)) // threadsperblock
        else:
            print('grid stride loops not implemented')
            exit()

        self.dev_output.copy_to_device(self.dev_x_space)
        dev_z = cuda.to_device(np.array([z]).astype(np.float64))

        self.fractal_kernel[blockspergrid, threadsperblock](self.dev_output, self.dev_y_space, dev_z)
        cuda.synchronize()

        self.dev_output.copy_to_host(self.output)
        lambda_min = np.amin(self.output)
        scaling_factor = np.amax(self.output) - lambda_min
        if (scaling_factor == 0):
            return np.zeros((self.size, self.size, 3))
        indexes = ((self.color_resolution-1)*(self.output-lambda_min) / scaling_factor).astype(int)
        gradient = self.get_gradient(indexes)
        image = gradient[indexes].reshape((self.size, self.size, 3))
        image = np.flip(image, axis=1)

        return image
    
    
    def create_fractal_video(self, num_frames, verbose=True):
        video = []
        for idx, z in enumerate(np.linspace(self.z_min, self.z_max, num_frames), 1):
            image = Image.fromarray(self.compute_fractal(z).astype(np.uint8)).convert('RGB')
            video.append(image)
            if verbose:
                print(f"frame {idx}/{num_frames}", end="\r")
        if verbose:
            print()
        return video


class FractalZoom(ComputeFractals):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.zoom_proportion = 0.98
    
    def get_mouse_coords_in_region(self, pos):
        pos_x = self.x_min + (self.x_max-self.x_min)*(pos[0]/self.size)
        pos_y = self.y_min + (self.y_max-self.y_min)*(1 - pos[1]/self.size) # 1 - is because 
        # y axis is pointing downwards in pygame
        return pos_x, pos_y
    
    def center_zoom(self, mouse_pos, coef):
        new_pos_x = coef*(mouse_pos[0] - self.size/2) + self.size/2
        new_pos_y = coef*(mouse_pos[1] - self.size/2) + self.size/2
        return new_pos_x, new_pos_y

    def zoom_to(self, pos, zoom_proportion):
        x_min = pos[0] - (zoom_proportion*(self.x_max - self.x_min))/2
        y_min = pos[1] - (zoom_proportion*(self.y_max - self.y_min))/2
        x_max = x_min + zoom_proportion*(self.x_max - self.x_min)
        y_max = y_min + zoom_proportion*(self.y_max - self.y_min)

        # boundary checks
        if (x_max - x_min) > 4:
            x_min = 0
            x_max = 4

        if (y_max - y_min) > 4:
            y_min = 0
            y_max = 4
             
        if (x_min < 0):
            x_max -= x_min
            x_min = 0

        if (y_min < 0):
            y_max -= y_min
            y_min = 0
        
        if (x_max > 4):
            x_min -= x_max - 4 
            x_max = 4

        if (y_max > 4):
            y_min -= y_max - 4 
            y_max = 4
        
        return x_min, x_max, y_min, y_max
    
    def run(self, z_interval):

        z = self.z_min

        FPS = 10

        pygame.init()
        
        clock = pygame.time.Clock()

        display = pygame.display.set_mode((self.size, self.size))

        curr_surf = pygame.surfarray.make_surface(self.compute_fractal(z))
        running = True

        new_image_event = pygame.USEREVENT + 1
        pygame.time.set_timer(new_image_event, 200)
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            
            mouse_buttons = pygame.mouse.get_pressed()
            keys = pygame.key.get_pressed()
    
            if keys[pygame.K_TAB]:
                z += z_interval
                z %= self.z_max

            elif mouse_buttons[0]:
                mouse_pos = pygame.mouse.get_pos()
                mouse_pos = self.center_zoom(mouse_pos, 0.1)
                mouse_pos = self.get_mouse_coords_in_region(mouse_pos)
                self.set_region(*self.zoom_to(mouse_pos, self.zoom_proportion))

            elif mouse_buttons[2]:
                mouse_pos = pygame.mouse.get_pos()
                mouse_pos = self.center_zoom(mouse_pos, 0.1)
                mouse_pos = self.get_mouse_coords_in_region(mouse_pos)
                self.set_region(*self.zoom_to(mouse_pos, 1/self.zoom_proportion))

            curr_surf = pygame.surfarray.make_surface(self.compute_fractal(z))

            display.blit(curr_surf, (0, 0))
            pygame.display.flip()
            clock.tick(FPS)
        pygame.quit()
