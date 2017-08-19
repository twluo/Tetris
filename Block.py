import pygame
import sys
import random
from enum import Enum
#block constants
BLOCK_HEIGHT = 16
BLOCK_WIDGTH = 16

#grid constants
GRID_WIDTH = 10
GRID_HEIGHT = 20

class Movement(Enum):
    MOVE_LEFT = 0
    MOVE_RIGHT = 1
    SOFT_DROP = 2
    HARD_DROP = 3
    ROTATE_LEFT = 4
    ROTATE_RIGHT = 5

class Tetromino(Enum):
    I_BLOCK = 1
    L_BLOCK = 2
    J_BLOCK = 3
    O_BLOCK = 4
    S_BLOCK = 5
    T_BLOCK = 6
    Z_BLOCK = 7

    def __repr__(self):
        return str(self.value)

class Color(Enum):
    I_BLOCK = (1, 0.5, 0.5)
    L_BLOCK = (0.5, 1, 0.5)
    J_BLOCK = (0.5, 0.5, 1)
    O_BLOCK = (0.8, 0.4, 0.8)
    S_BLOCK = (0.8, 0.8, 0.4)
    T_BLOCK = (0.4, 0.8, 0.8)
    Z_BLOCK = (0.7, 0.7, 0.7)

class Block(object):

    def __init__(self, tetromino_id):
        #16 x 16
        self.image = pygame.image.load("block.png")
        self.image.convert_alpha()
        self.color_surface(self.image, Color[tetromino_id.name].value)


    def draw(self, screen, center):
        origin = (center[0] - self.image.get_width()/2, center[1] - self.image.get_height()/2)
        screen.blit(self.image, origin)

    def color_surface(self, surface, newColor):
        newColor = [x * 255 for x in newColor]
        self.image.fill(newColor, None, pygame.BLEND_RGBA_MULT)

def clamp(value, min_value, max_value):
    return max(min_value, min(value, max_value))

class BasicTetromino(object):

    def __init__(self, tetromino_id):
        self.tetromino_id = tetromino_id
        self.rotation = 0
        self.configurations = [[(0, 0) for x in range(5) for x in range(5)]]
        self.position = (4, -3)
        self.moves = {
            Movement.MOVE_LEFT : self.move_left,
            Movement.MOVE_RIGHT : self.move_right,
            Movement.SOFT_DROP : self.soft_drop,
            Movement.ROTATE_LEFT : self.rotate_left,
            Movement.ROTATE_RIGHT : self.rotate_right,
            Movement.HARD_DROP : self.hard_drop
        }

    def move(self, movement, grid):
        self.moves[movement](grid)

    def rotate_left(self, grid):
        rotation = (self.rotation - 1) % 4
        if self.check_collision(self.position, rotation, grid):
            self.rotation = rotation

    def rotate_right(self, grid):
        rotation = (self.rotation + 1) % 4
        if self.check_collision(self.position, rotation, grid):
            self.rotation = rotation

    def move_right(self, grid):
        position = (clamp(self.position[0] + 1, 0, 9), self.position[1])
        if self.check_collision(position, self.rotation, grid):
            self.position = position


    def move_left(self, grid):
        position = (clamp(self.position[0] - 1, 0, 9), self.position[1])
        if self.check_collision(position, self.rotation, grid):
            self.position = position


    def soft_drop(self, grid):
        position = self.position[0], self.position[1] + 1
        if self.check_collision(position, self.rotation, grid):
            self.position = position

    def hard_drop(self, grid):
        for y_position in range(self.position[1], 21):
            curr_position = (self.position[0], y_position)
            next_position = (self.position[0], y_position + 1)
            self.position = curr_position
            if not self.check_collision(next_position, self.rotation, grid):
                break

    def check_collision(self, position, rotation, grid):
        for x_offset, y_offset in self.configurations[rotation]:
            x = position[1] + y_offset
            y = position[0] + x_offset
            if x < GRID_HEIGHT and y >= 0 and y < GRID_WIDTH:
                if grid[x][y] is not 0 and x >= 0:
                    print("Collided", x, y)
                    return False
            else:
                print("Out of bounds")
                return False
        else:
            return True

    def is_game_over(self, grid):
        return not self.check_collision(self.position, self.rotation, grid)

class ITetromino(BasicTetromino):

    def __init__(self):
        super(ITetromino, self).__init__(Tetromino.I_BLOCK)
        self.position = (4, -2)
        self.configurations = [[(0, 0), (-1, 0), (1, 0), (2, 0)],
                               [(0, 0), (0, 1), (0, -1), (0, -2)],
                               [(0, 0), (1, 0), (-1, 0), (-2, 0)],
                               [(0, 0), (0, -1), (0, 1), (0, 2)]]    

class LTetromino(BasicTetromino):

    def __init__(self):
        super(LTetromino, self).__init__(Tetromino.L_BLOCK)
        self.configurations = [[(0, 0), (-1, 0), (1, 0), (1, -1)],
                               [(0, 0), (0, 1), (0, -1), (1, 1)],
                               [(0, 0), (1, 0), (-1, 0), (-1, 1)],
                               [(0, 0), (0, 1), (0, -1), (-1, -1)]]

class JTetromino(BasicTetromino):

    def __init__(self):
        super(JTetromino, self).__init__(Tetromino.J_BLOCK)
        self.configurations = [[(0, 0), (-1, 0), (1, 0), (-1, -1)],
                               [(0, 0), (0, 1), (0, -1), (1, -1)],
                               [(0, 0), (1, 0), (-1, 0), (1, 1)],
                               [(0, 0), (0, 1), (0, -1), (-1, 1)]]

class OTetromino(BasicTetromino):

    def __init__(self):
        super(OTetromino, self).__init__(Tetromino.O_BLOCK)
        self.configurations = [[(0, 0), (0, 1), (1, 1), (1, 0)] for x in range(5)]

class STetromino(BasicTetromino):

    def __init__(self):
        super(STetromino, self).__init__(Tetromino.S_BLOCK)
        self.configurations = [[(0, 0), (-1, 0), (0, -1), (1, -1)],
                               [(0, 0), (0, -1), (1, 0), (1, 1)],
                               [(0, 0), (1, 0), (0, 1), (-1, 1)],
                               [(0, 0), (0, 1), (-1, 0), (-1, -1)]]

class TTetromino(BasicTetromino):

    def __init__(self):
        super(TTetromino, self).__init__(Tetromino.T_BLOCK)
        self.configurations = [[(0, 0), (-1, 0), (0, -1), (1, 0)],
                               [(0, 0), (0, 1), (1, 0), (0, -1)],
                               [(0, 0), (1, 0), (0, 1), (-1, 0)],
                               [(0, 0), (0, -1), (-1, 0), (0, 1)]]

                           
class ZTetromino(BasicTetromino):

    def __init__(self):
        super(ZTetromino, self).__init__(Tetromino.Z_BLOCK)
        self.configurations = [[(0, 0), (1, 0), (0, -1), (-1, -1)],
                               [(0, 0), (0, 1), (1, 0), (1, -1)],
                               [(0, 0), (-1, 0), (0, 1), (1, 1)],
                               [(0, 0), (0, -1), (-1, 0), (-1, 1)]]

class Grid(object):
    def __init__(self):
        self.grid = [[0 for x in range(GRID_WIDTH)] for x in range(GRID_HEIGHT)]
        self.size = (GRID_WIDTH * BLOCK_WIDGTH, GRID_HEIGHT * BLOCK_HEIGHT)
        self.center = (255, 255)
        self.seven_bag = self.shuffle()
        self.active_tetromino = None

    def shuffle(self):
        seven_bag = [k for k in Tetromino]
        print(seven_bag)
        random.shuffle(seven_bag)
        return seven_bag

    def pretty_print(self):
        for x in self.grid:
            print(x.value) if x in Tetromino else print(x)

    def spawn_tetromino(self):
        if len(self.seven_bag) is 0:
            self.seven_bag = self.shuffle()
        new_piece = self.seven_bag[0]
        if new_piece is Tetromino.I_BLOCK:
            self.active_tetromino = ITetromino()
        elif new_piece is Tetromino.J_BLOCK:
            self.active_tetromino = JTetromino()
        elif new_piece is Tetromino.L_BLOCK:
            self.active_tetromino = LTetromino()
        elif new_piece is Tetromino.O_BLOCK:
            self.active_tetromino = OTetromino()
        elif new_piece is Tetromino.S_BLOCK:
            self.active_tetromino = STetromino()
        elif new_piece is Tetromino.T_BLOCK:
            self.active_tetromino = TTetromino()
        elif new_piece is Tetromino.Z_BLOCK:
            self.active_tetromino = ZTetromino()
        if self.active_tetromino.is_game_over(self.grid):
            print("Game Over")
        self.seven_bag = self.seven_bag[1:]
        self.update_grid()

    def update_grid(self):
        for x_offset, y_offset in self.active_tetromino.configurations[self.active_tetromino.rotation]:
            x = self.active_tetromino.position[1] + y_offset
            y = self.active_tetromino.position[0] + x_offset
            if x >= 0 and x <= GRID_HEIGHT and y >= 0 and y <= GRID_WIDTH:
                self.grid[x][y] = self.active_tetromino.tetromino_id
        self.pretty_print()

    def clear_active_tetromino(self):
        for x_offset, y_offset in self.active_tetromino.configurations[self.active_tetromino.rotation]:
            x = self.active_tetromino.position[1] + y_offset
            y = self.active_tetromino.position[0] + x_offset
            if x >= 0 and x <= GRID_HEIGHT and y >= 0 and y <= GRID_WIDTH:
                self.grid[x][y] = 0

    def move_active_tetromino(self, movement):
        self.clear_active_tetromino()
        self.active_tetromino.move(movement, self.grid)
        self.update_grid()
        print(self.check_and_clear_lines())

    def check_and_clear_lines(self):
        counter = 0
        for row in self.grid:
            for block in row:
                if block is 0:
                    break
            else:
                self.grid.remove(row)
                self.grid.insert(0, [0 for x in range(GRID_WIDTH)])
                counter += 1
        return counter

    def draw(self, screen):
        for x in range(GRID_WIDTH):
            for y in range(GRID_HEIGHT):
                if self.grid[y][x] is not 0:
                    block = Block(Tetromino(self.grid[y][x]))
                    origin_x = self.center[1] - (self.size[0] - BLOCK_HEIGHT * (2 * x + 1))/2
                    origin_y = self.center[0] - (self.size[1] - BLOCK_WIDGTH * (2 * y + 1 ))/2
                    block.draw(screen, (origin_x, origin_y))

class GameController(object):
    def handleEvents(self):
        event = pygame.event.poll()
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit(0)
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_LEFT:
                self.grid.move_active_tetromino(Movement.MOVE_LEFT)
            if event.key == pygame.K_RIGHT:
                self.grid.move_active_tetromino(Movement.MOVE_RIGHT)
            if event.key == pygame.K_DOWN:
                self.grid.move_active_tetromino(Movement.SOFT_DROP)
            if event.key == pygame.K_z:
                self.grid.move_active_tetromino(Movement.ROTATE_LEFT)
            if event.key == pygame.K_x:
                self.grid.move_active_tetromino(Movement.ROTATE_RIGHT)
            if event.key == pygame.K_SPACE:
                self.grid.move_active_tetromino(Movement.HARD_DROP)
                self.grid.spawn_tetromino()


    def main(self):
        pygame.init()
        screen = pygame.display.set_mode((500, 500))
        self.grid = Grid()
        self.grid.spawn_tetromino()
        self.grid.update_grid()
        while True:
            self.handleEvents()
            screen.fill((0,0,0))
            self.grid.draw(screen)
            pygame.display.update()


if __name__ == '__main__':
    game_controller = GameController()
    game_controller.main()
    