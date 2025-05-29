import pygame
from lyapunov_core import *

class FractalZoom(ComputeFractals):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.zoom_proportion = 0.98
    
    def get_mouse_coords_in_region(self, pos):
        pos_x = self.x_min + (self.x_max-self.x_min)*(pos[0]/self.size)
        pos_y = self.y_min + (self.y_max-self.y_min)*(1 - pos[1]/self.size)
        # 1 - is because y axis is pointing downwards in pygame
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
                                size=2000, colors=self.colors, color_resolution=1900, num_iter=8000
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

def main():
    fractal_zoom = FractalZoom(z_min = 2.86, pattern = "yxzzyxyzxyxzyzxyzxyxzyxzyxzyyzx", size = 500,
                               x_min = 1.936, x_max = 2.319, y_min = 3.617, y_max = 4,
                               colors = ColorPalettes.black_purple, color_resolution=1800)
    fractal_zoom.run(0.01)

if __name__ == "__main__":
    main()