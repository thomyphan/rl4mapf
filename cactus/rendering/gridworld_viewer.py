import pygame

BLACK = (0, 0, 0)
DARK_GRAY = (125, 125, 125)
GRAY = (175, 175, 175)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
ORANGE = (255, 150, 0)
BLUE = (0, 0, 255)
LIGHT_BLUE = (51, 226, 253)
GREEN = (0, 255, 0)
MAGENTA = (255, 0, 255)
MAROON = (128, 0, 0)
CYAN = (0, 255, 255)
TEAL = (0, 128, 128)
PURPLE = (128, 0, 128)

AGENT_COLORS = [RED, BLUE, ORANGE, MAGENTA, PURPLE, TEAL, MAROON, GREEN, DARK_GRAY, CYAN]

class GridworldViewer:
    def __init__(self, width, height, cell_size=10, fps=30):
        pygame.init()
        self.cell_size = cell_size
        self.width = cell_size*width
        self.height = cell_size*height
        self.clock = pygame.time.Clock()
        self.fps = fps
        pygame.display.set_caption("MAPF Environment")
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.event.set_blocked(pygame.MOUSEMOTION)  # we do not need mouse movement events

    def agent_color(self, agent_id):
        nr_colors = len(AGENT_COLORS)
        return AGENT_COLORS[agent_id%nr_colors]
              
    def draw_state(self, env):
        self.screen.fill(BLACK)
        for x in range(env.rows):
            for y in range(env.columns):
                if not env.obstacle_map[x][y]:
                    self.draw_pixel(x, y, WHITE)
                agent_id = env.occupied_goal_positions[x][y]
                if agent_id >= 0:
                    self.draw_pixel(x, y, self.agent_color(agent_id))
                agent_id = env.current_position_map[x][y]
                if agent_id >= 0:
                    self.draw_circle(x, y, self.agent_color(agent_id))
        pygame.display.flip()
        self.clock.tick(self.fps)
        return self.check_for_interrupt()
    
    def draw_pixel(self, x, y, color):
        pygame.draw.rect(self.screen, color,
                        pygame.Rect(
                            x * self.cell_size+1,
                            y * self.cell_size+1,
                            self.cell_size-2,
                            self.cell_size-2),
                        0)
    
    def draw_circle(self, x, y, color):
        radius = int(self.cell_size/2)
        center_x = x * self.cell_size + radius
        center_y = y * self.cell_size + radius
        center = (center_x, center_y)
        pygame.draw.circle(self.screen, color, center, radius-2)

    def check_for_interrupt(self):
        key_state = pygame.key.get_pressed()
        for event in pygame.event.get():
            if event.type == pygame.QUIT or key_state[pygame.K_ESCAPE]:
                return True
        return False

    def close(self):
        pygame.quit()

def render(env, viewer):
    if viewer is None:
        viewer = GridworldViewer(env.columns, env.rows)
    viewer.draw_state(env)
    return viewer
