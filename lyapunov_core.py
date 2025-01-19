import numpy as np
import pygame
from collections import Counter
from math import log
from numba import cuda
from PIL import Image
from bisect import bisect_left
from dataclasses import dataclass

@dataclass
class ColorPalettes:
    yellow_purple_black = ['#E0D12B', '#DAA520', '#FFBF00', '#FF9500', '#FFA500', '#FF7F50',
        '#FA8072', '#F08080', '#FFB6C1', "#E0BE4C", "#F5CAAF", '#DA70D6', '#BA55D3', '#9370DB',
        '#8A2BE2', '#6A0DAD', '#4B0082', '#2E0854', '#1A0028', '#000000']

    black_red_blue = ["#03071e", "#370617", "#6a040f", "#9d0208", "#d00000", "#FF7F50",
        "#B8860B", "#FFC700", "#bdb76b", "#6b8e23", "#556b2f", '#b3cde0', '#a1c4d6',
        '#8bbdd9', '#7aaed4', '#699ecf', '#5e94c9', '#4b82b4']    

    purple_red_blue = ["#411d31", "#631b34", "#32535f", "#0b8a8f", "#0eaf9b", "#30e1b9"]

    black_magenta_purple = ["#000000", "#b80049", "#ea569e", "#ffa653", "#fbe7b5", "#ff89dc",
        "#bb19e1", "#4a17a1", "#071c5a"]

    red_blue_red = ["#401b20", "#8e252e", "#9350aa", "#0e3abf", "#24793d", "#ffab89",
        "#fc4e51", "#de024e"]

    black_purple = ["#130208", "#1f0510", "#31051e", "#460e2b", "#7c183c", "#d53c6a",
        "#ff8274"]

    black_orange_yellow = ["#202215", "#3a2802", "#963c3c", "#ca5a2e", "#ff7831",
        "#f39949", "#ebc275", "#dfd785"]

    blue_gray_pink = ["#292831", "#333f58", "#4a7a96", "#ee8695", "#fbbbad"]

    red_blue_black = ["#de024e", "#fc4e51", "#ffab89", "#24793d", "#0e3abf",
        "#9350aa", "#8e252e", "#401b20"]

    red_yellow_blue = ["#ee4035", "#f37736", "#fdf498", "#7bc043", "#0392cf", "#8409da"]

    black_green_orange = ["#000000", "#003300", "#006600", "#CC6600", "#993300"]

    red_orange_yellow = ["#660000", "#990000", "#CC3333", "#FF9900", "#FFC333", "#CCFFCC"]

    orange_blue_black = ["#9c2a0b", "#ab2d0a", "#bd350b", "#bd400b", "#bd4f0b", "#cc760c",
        "#cc7f0c", "#d9910b", "#d9ab16", "#4ab80f", "#0f61b8", "#0a2ba3", "#09188f", 
        "#081680", "#060f57", "#020733", "#00010a"]
    
    black_red_green = ["#03071e", "#370617", "#6a040f", "#9d0208", "#d00000","#FF7F50",
        "#B8860B", "#FFC700", "#bdb76b", "#6b8e23", "#556b2f", "#006600", "#004d00"]


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

        self.set_pattern(pattern)

        y_space = np.tile(np.linspace(self.y_min, self.y_max, self.size), self.size).astype(np.float64)
        self.dev_y_space = cuda.to_device(y_space)

        x_space = np.repeat(np.linspace(self.x_min, self.x_max, self.size), self.size).astype(np.float64)
        self.dev_x_space = cuda.to_device(x_space)

        self.dev_output = cuda.device_array_like(self.dev_x_space)

        self.output = np.zeros_like(x_space)

        self.gpu = cuda.get_current_device()

        print(f"used GPU: ", self.gpu.name.decode("utf-8"))

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

    def compute_fractal(self, z, verbose = False):

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

        if (verbose):
            print("copied data to GPU, executing fractal kernel")

        self.fractal_kernel[blockspergrid, threadsperblock](self.dev_output, self.dev_y_space, dev_z)
        cuda.synchronize()

        if (verbose):
            print("fractal computed, copying data back to cpu")

        self.dev_output.copy_to_host(self.output)

        if (verbose):
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
            x_min = 0.01
            x_max = 4

        if (y_max - y_min) > 4:
            y_min = 0.01
            y_max = 4
             
        if (x_min < 0):
            x_max -= x_min - 0.01
            x_min = 0.01

        if (y_min < 0):
            y_max -= y_min - 0.01
            y_min = 0.01
        
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
                    
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE or event.key == pygame.K_s:

                        print(f"pattern : {self.pattern}")
                        print(f"x_min : {self.x_min}")
                        print(f"x_max : {self.x_max}")
                        print(f"y_min : {self.y_min}")
                        print(f"y_max : {self.y_max}")
                        print(f"z : {z}")

                        if event.key == pygame.K_s:
                            fractal_computer = ComputeFractals(
                                pattern=self.pattern, 
                                x_min = self.x_min, x_max = self.x_max, 
                                y_min = self.y_min, y_max = self.y_max,
                                size=2000, color_resolution=1900, num_iter=8000
                            )
                            
                            image = fractal_computer.compute_fractal(z)

                            image = Image.fromarray(np.swapaxes(image.astype(np.uint8), 0, 1))
        
                            image.show()

                            while (not any(pygame.mouse.get_pressed())):
                                pygame.event.get()
                                if (pygame.key.get_pressed()[pygame.K_s]):        
                                    image.save(str(self.pattern) + '_' + 
                                            str(round(self.x_min, 3)) + '_' + 
                                            str(round(self.x_max, 3)) + '_' + 
                                            str(round(self.y_min, 3)) + '_' + 
                                            str(round(self.y_max, 3)) + '_z_' +
                                            str(round(z, 3)) + '.png')
                                    
                                    print("image saved!")
                                    break
                                display.blit(curr_surf, (0, 0))
                                pygame.display.flip()
                                clock.tick(FPS)

                    if event.key == pygame.K_c:
                        self.set_pattern(self.pattern[-1] + self.pattern[:-1])
            
            mouse_buttons = pygame.mouse.get_pressed()
            keys = pygame.key.get_pressed()
    
            if keys[pygame.K_TAB]:
                z += z_interval
                z %= self.z_max
            elif keys[pygame.K_BACKSPACE]:
                z -= z_interval
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
