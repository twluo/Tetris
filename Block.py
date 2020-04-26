import pygame
import sys
import random
from enum import Enum

# RENDERING CONSTANTS
SCALING_FACTOR = 2
WINDOW_HEIGHT = 500 * SCALING_FACTOR
WINDOW_WIDTH = 500 * SCALING_FACTOR
BLOCK_HEIGHT = 16 * SCALING_FACTOR
BLOCK_WIDTH = 16 * SCALING_FACTOR

# GAME CONSTANTS
GRID_WIDTH = 10
GRID_HEIGHT = 20
TETROMINO_BOX_WIDTH = 5
TETROMINO_BOX_HEIGHT = 5

# GAMEPLAY CONSTANTS
GRAVITY_DELAY = 1000
LOCK_DELAY = 1000
DAS = 100
ARR = 30
NEXT_LENGTH = 3


def check_bounds(x, y):
    return 0 <= x <= GRID_HEIGHT and 0 <= y <= GRID_WIDTH


def spawn_bag():
    seven_bag = [k for k in Tetromino]
    random.shuffle(seven_bag)
    return seven_bag


def clamp(value, min_value, max_value):
    return max(min_value, min(value, max_value))


def tupple_add(tupple1, tupple2):
    return tupple1[0] + tupple2[0], tupple1[1] + tupple2[1]


def tupple_sub(tupple1, tupple2):
    return tupple1[0] - tupple2[0], tupple1[1] - tupple2[1]


def get_tetromino(tetromino_id):
    if tetromino_id is Tetromino.I_BLOCK:
        return ITetromino()
    elif tetromino_id is Tetromino.J_BLOCK:
        return JTetromino()
    elif tetromino_id is Tetromino.L_BLOCK:
        return LTetromino()
    elif tetromino_id is Tetromino.O_BLOCK:
        return OTetromino()
    elif tetromino_id is Tetromino.S_BLOCK:
        return STetromino()
    elif tetromino_id is Tetromino.T_BLOCK:
        return TTetromino()
    elif tetromino_id is Tetromino.Z_BLOCK:
        return ZTetromino()


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
        # 16 x 16
        self.image = pygame.image.load("block.png")
        self.image = pygame.transform.scale(self.image, (BLOCK_HEIGHT, BLOCK_WIDTH))
        self.image.convert_alpha()
        self.color_surface(Color[tetromino_id.name].value)

    def draw(self, screen, center):
        origin = (center[0] - self.image.get_width()/2, center[1] - self.image.get_height()/2)
        screen.blit(self.image, origin)

    def color_surface(self, new_color):
        new_color = [x * 255 for x in new_color]
        self.image.fill(new_color, None, pygame.BLEND_RGBA_MULT)


class BasicTetromino(object):

    def __init__(self, tetromino_id):
        self.tetromino_id = tetromino_id
        self.rotation = 0
        self.configurations = [[(0, 0) for x in range(TETROMINO_BOX_WIDTH) for x in range(TETROMINO_BOX_HEIGHT)]]
        self.position = (4, -3)
        self.moves = {
            Movement.MOVE_LEFT: self.move_left,
            Movement.MOVE_RIGHT: self.move_right,
            Movement.SOFT_DROP: self.soft_drop,
            Movement.ROTATE_LEFT: self.rotate_left,
            Movement.ROTATE_RIGHT: self.rotate_right,
            Movement.HARD_DROP: self.hard_drop
        }

    def move(self, movement, grid):
        return self.moves[movement](grid)

    def rotate_helper(self, rot):
        if self.tetromino_id is Tetromino.O_BLOCK:
            if rot is 0:
                return [(0, 0)]
            elif rot is 1:
                return [(0, 1)]
            elif rot is 2:
                return [(-1, 1)]
            else:
                return [(-1, 0)]
        elif self.tetromino_id is Tetromino.I_BLOCK:
            if rot is 0:
                return [(0, 0), (-1, 0), (2, 0), (-1, 0), (2, 0)]
            elif rot is 1:
                return [(-1, 0), (0, 0), (0, 0), (0, -1), (0, 2)]
            elif rot is 2:
                return [(-1, -1), (1, -1), (-2, -1), (1, 0), (-2, 0)]
            else:
                return [(0, -1), (0, -1), (0, -1), (0, 1), (0, -2)]
        else:
            if rot is 0:
                return [(0, 0), (0, 0), (0, 0), (0, 0), (0, 0)]
            elif rot is 1:
                return [(0, 0), (1, 0), (1, 1), (0, -2), (1, -2)]
            elif rot is 2:
                return [(0, 0), (0, 0), (0, 0), (0, 0), (0, 0)]
            else:
                return [(0, 0), (-1, 0), (-1, 1), (0, -2), (-1, -2)]

    def rotate_left(self, grid):
        new_rotation = (self.rotation - 1) % 4
        rotate_flag, new_position = self.rotate(grid, self.rotation, new_rotation)
        if rotate_flag:
            self.position = new_position
            self.rotation = new_rotation
            return True
        return False

    def rotate_right(self, grid):
        new_rotation = (self.rotation + 1) % 4
        rotate_flag, new_position = self.rotate(grid, self.rotation, new_rotation)
        if rotate_flag:
            self.position = new_position
            self.rotation = new_rotation
            return True
        return False

    def rotate(self, grid, old_rot, new_rot):
        old_offset = self.rotate_helper(old_rot)
        curr_offset = self.rotate_helper(new_rot)
        for old, curr in zip(old_offset, curr_offset):
            offset = tupple_sub(old, curr)
            new_position = tupple_add(self.position, offset)
            if self.check_collision(new_position, new_rot, grid):
                return True, new_position
        return False, self.position

    def move_right(self, grid):
        position = (clamp(self.position[0] + 1, 0, 9), self.position[1])
        if self.check_collision(position, self.rotation, grid):
            self.position = position
            return True
        return False

    def move_left(self, grid):
        position = (clamp(self.position[0] - 1, 0, 9), self.position[1])
        if self.check_collision(position, self.rotation, grid):
            self.position = position
            return True
        return False

    def soft_drop(self, grid):
        position = self.position[0], self.position[1] + 1
        if self.check_collision(position, self.rotation, grid):
            self.position = position
            return True
        return False

    def hard_drop(self, grid):
        for y_position in range(self.position[1], GRID_HEIGHT + 1):
            curr_position = (self.position[0], y_position)
            next_position = (self.position[0], y_position + 1)
            if not self.check_collision(next_position, self.rotation, grid):
                self.position = curr_position
                return True
        return False

    def check_collision(self, position, rotation, grid):
        for x_offset, y_offset in self.configurations[rotation]:
            x = position[1] + y_offset
            y = position[0] + x_offset
            if x < GRID_HEIGHT and 0 <= y < GRID_WIDTH:
                if grid[x][y] is not 0 and x >= 0:
                    return False
            else:
                return False
        else:
            return True

    def is_game_over(self, grid):
        return not self.check_collision((4, 0), self.rotation, grid)

    def __str__(self):
        return str(self.tetromino_id)

    def __repr__(self):
        return str(self.tetromino_id)


class ITetromino(BasicTetromino):

    def __init__(self):
        super(ITetromino, self).__init__(Tetromino.I_BLOCK)
        self.position = (4, -2)
        self.configurations = [[(0, 0), (-1, 0), (1, 0), (2, 0)],
                               [(0, 0), (0, -1), (0, 1), (0, 2)],
                               [(0, 0), (1, 0), (-1, 0), (-2, 0)],
                               [(0, 0), (0, -1), (0, 1), (0, -2)]]


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
        self.configurations = [[(0, 0), (0, -1), (1, -1), (1, 0)],
                               [(0, 0), (1, 1), (0, 1), (1, 0)],
                               [(0, 0), (-1, 1), (0, 1), (-1, 0)],
                               [(0, 0), (-1, -1), (0, -1), (-1, 0)]]


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
        self.buffer = WINDOW_WIDTH/50
        self.grid = [[0 for x in range(GRID_WIDTH)] for x in range(GRID_HEIGHT)]
        self.grid_size = (GRID_WIDTH * BLOCK_WIDTH, GRID_HEIGHT * BLOCK_HEIGHT)
        self.grid_center = (WINDOW_WIDTH/2, WINDOW_HEIGHT/2)
        self.grid_top_left = (self.grid_center[0] - self.grid_size[0] / 2, self.grid_center[1] - self.grid_size[1] / 2)
        self.hold_size = ((TETROMINO_BOX_WIDTH + 1) * BLOCK_WIDTH, (TETROMINO_BOX_HEIGHT - 1) * BLOCK_HEIGHT)
        self.hold_center = (self.grid_center[0] - self.grid_size[0]/2 - self.buffer - self.hold_size[0]/2,
                            self.grid_center[1] - self.grid_size[1]/2 + self.hold_size[1]/2)
        self.hold_top_left = (self.hold_center[0] - self.hold_size[0] / 2, self.hold_center[1] - self.hold_size[1] / 2)
        self.next_size = ((TETROMINO_BOX_WIDTH + 1) * BLOCK_WIDTH,
                          ((TETROMINO_BOX_HEIGHT - 1) * NEXT_LENGTH + TETROMINO_BOX_HEIGHT/2) * BLOCK_HEIGHT)
        self.next_center = (self.grid_center[0] + self.grid_size[0]/2 + self.buffer + self.next_size[0]/2,
                            self.grid_center[1] - self.grid_size[1]/2 + self.next_size[1]/2)
        self.next_top_left = (self.next_center[0] - self.next_size[0] / 2, self.next_center[1] - self.next_size[1] / 2)
        self.seven_bag = spawn_bag()
        self.next_bag = spawn_bag()
        self.active_tetromino = None
        self.next_tetrominos = [get_tetromino(self.seven_bag.pop(0)) for x in range(NEXT_LENGTH)]
        self.gravity_timer = 0
        self.lock_timer = 0
        self.game_over = False
        self.hold_flag = False
        self.hold_tetromino = None

    def pretty_print(self):
        for x in self.grid:
            print(x.value) if x in Tetromino else print(x)

    def swap_tetromino(self):
        if self.hold_flag is True:
            return False
        self.hold_flag = True
        self.clear_active_tetromino()
        if self.hold_tetromino is None:
            self.hold_tetromino = get_tetromino(self.active_tetromino.tetromino_id)
            self.spawn_new_tetromino()
            return True
        new_tetromino = self.hold_tetromino.tetromino_id
        self.hold_tetromino = get_tetromino(self.active_tetromino.tetromino_id)
        self.spawn_tetromino(new_tetromino)
        return True

    def spawn_new_tetromino(self):
        self.spawn_tetromino(self.next_tetrominos.pop(0).tetromino_id)
        self.next_tetrominos.append(get_tetromino(self.seven_bag.pop(0)))
        self.seven_bag.append(self.next_bag.pop(0))
        if len(self.next_bag) is 0:
            self.next_bag = spawn_bag()
        self.hold_flag = False

    def spawn_tetromino(self, new_tetromino):
        self.active_tetromino = get_tetromino(new_tetromino)
        self.gravity_timer = pygame.time.get_ticks()
        self.lock_timer = pygame.time.get_ticks()
        self.update_grid()
        if self.active_tetromino.is_game_over(self.grid):
            print("Game Over")
            self.game_over = True

    def apply_gravity(self):
        curr_time = pygame.time.get_ticks()
        if curr_time - self.gravity_timer >= GRAVITY_DELAY:
            self.move_active_tetromino(Movement.SOFT_DROP)
            self.gravity_timer = curr_time

    def apply_locking(self):
        curr_time = pygame.time.get_ticks()
        if curr_time - self.lock_timer >= LOCK_DELAY:
            self.move_active_tetromino(Movement.HARD_DROP)
            self.check_and_clear_lines()
            self.spawn_new_tetromino()

    def update_grid(self):
        for x_offset, y_offset in self.active_tetromino.configurations[self.active_tetromino.rotation]:
            x = self.active_tetromino.position[1] + y_offset
            y = self.active_tetromino.position[0] + x_offset
            if check_bounds(x, y):
                self.grid[x][y] = self.active_tetromino.tetromino_id
        # self.pretty_print()

    def clear_active_tetromino(self):
        for x_offset, y_offset in self.active_tetromino.configurations[self.active_tetromino.rotation]:
            x = self.active_tetromino.position[1] + y_offset
            y = self.active_tetromino.position[0] + x_offset
            if check_bounds(x, y):
                self.grid[x][y] = 0

    def move_active_tetromino(self, movement):
        self.clear_active_tetromino()
        has_it_moved = self.active_tetromino.move(movement, self.grid)
        self.update_grid()
        if has_it_moved:
            self.gravity_timer = pygame.time.get_ticks()
            self.lock_timer = pygame.time.get_ticks()

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
        pygame.draw.rect(screen, (255, 255, 255), (self.grid_top_left, self.grid_size), 1)
        for x in range(GRID_WIDTH):
            for y in range(GRID_HEIGHT):
                if self.grid[y][x] is not 0:
                    block = Block(Tetromino(self.grid[y][x]))
                    origin_x = self.grid_center[1] - (self.grid_size[0] - BLOCK_HEIGHT * (2 * x + 1)) / 2
                    origin_y = self.grid_center[0] - (self.grid_size[1] - BLOCK_WIDTH * (2 * y + 1)) / 2
                    block.draw(screen, (origin_x, origin_y))

        if self.active_tetromino.position[1] <= 3:
            block = Block(Tetromino(self.active_tetromino.tetromino_id))
            for x_offset, y_offset in self.active_tetromino.configurations[self.active_tetromino.rotation]:
                y = self.active_tetromino.position[1] + y_offset
                x = self.active_tetromino.position[0] + x_offset
                origin_x = self.grid_center[1] - (self.grid_size[0] - BLOCK_HEIGHT * (2 * x + 1)) / 2
                origin_y = self.grid_center[0] - (self.grid_size[1] - BLOCK_WIDTH * (2 * y + 1)) / 2
                block.draw(screen, (origin_x, origin_y))

        pygame.draw.rect(screen, (255, 255, 255), (self.hold_top_left, self.hold_size), 1)
        if self.hold_tetromino is not None:
            block = Block(Tetromino(self.hold_tetromino.tetromino_id))
            for x_offset, y_offset in self.hold_tetromino.configurations[0]:
                y = 2 + y_offset
                x = 2 + x_offset
                origin_x = self.hold_center[0] - (self.hold_size[0] - BLOCK_HEIGHT * (2 * x + 1)) / 2
                origin_y = self.hold_center[1] - (self.hold_size[1] - BLOCK_WIDTH * (2 * y + 1)) / 2
                block.draw(screen, (origin_x, origin_y))

        pygame.draw.rect(screen, (255, 255, 255), (self.next_top_left, self.next_size), 1)
        for i in range(len(self.next_tetrominos)):
            block = Block(Tetromino(self.next_tetrominos[i].tetromino_id))
            for x_offset, y_offset in self.next_tetrominos[i].configurations[0]:
                y = 2 + y_offset + TETROMINO_BOX_HEIGHT * i
                x = 2 + x_offset
                origin_x = self.next_center[0] - (self.next_size[0] - BLOCK_HEIGHT * (2 * x + 1)) / 2
                origin_y = self.next_center[1] - (self.next_size[1] - BLOCK_WIDTH * (2 * y + 1)) / 2
                block.draw(screen, (origin_x, origin_y))


class GameController(object):
    def __init__(self):
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        self.grid = Grid()
        self.grid.spawn_new_tetromino()
        self.grid.update_grid()
        self.move_left = False
        self.move_right = False
        self.move_down = False
        self.move_left_das_timer = pygame.time.get_ticks()
        self.move_left_arr_timer = pygame.time.get_ticks()
        self.move_right_das_timer = pygame.time.get_ticks()
        self.move_right_arr_timer = pygame.time.get_ticks()
        self.move_down_arr_timer = pygame.time.get_ticks()

    def handle_events(self):
        event = pygame.event.poll()
        if event.type == pygame.QUIT or self.grid.game_over:
            pygame.quit()
            sys.exit(0)
        keys = pygame.key.get_pressed()
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_a:
                self.move_left = True
                self.move_right = False
                self.grid.move_active_tetromino(Movement.MOVE_LEFT)
                self.move_left_das_timer = pygame.time.get_ticks()
                self.move_left_arr_timer = pygame.time.get_ticks()
            if event.key == pygame.K_d:
                self.move_right = True
                self.move_left = False
                self.grid.move_active_tetromino(Movement.MOVE_RIGHT)
                self.move_right_das_timer = pygame.time.get_ticks()
                self.move_right_arr_timer = pygame.time.get_ticks()
            if event.key == pygame.K_s:
                self.move_down = True
                self.grid.move_active_tetromino(Movement.SOFT_DROP)
            if event.key == pygame.K_COMMA:
                self.grid.move_active_tetromino(Movement.ROTATE_LEFT)
            if event.key == pygame.K_PERIOD:
                self.grid.move_active_tetromino(Movement.ROTATE_RIGHT)
            if event.key == pygame.K_SPACE:
                self.grid.swap_tetromino()
            if event.key == pygame.K_w:
                self.grid.move_active_tetromino(Movement.HARD_DROP)
                self.grid.spawn_new_tetromino()
                self.grid.check_and_clear_lines()

        if event.type == pygame.KEYUP:
            if event.key == pygame.K_a:
                self.move_left = False
                if keys[pygame.K_d]:
                    self.move_right = True
            if event.key == pygame.K_d:
                self.move_right = False
                if keys[pygame.K_a]:
                    self.move_left = True
            if event.key == pygame.K_s:
                self.move_down = False

        if self.move_left:
            curr_timer = pygame.time.get_ticks()
            if curr_timer - self.move_left_das_timer > DAS:
                if curr_timer - self.move_left_arr_timer > ARR:
                    self.grid.move_active_tetromino(Movement.MOVE_LEFT)
                    self.move_left_arr_timer = curr_timer

        if self.move_right:
            curr_timer = pygame.time.get_ticks()
            if curr_timer - self.move_right_das_timer > DAS:
                if curr_timer - self.move_right_arr_timer > ARR:
                    self.grid.move_active_tetromino(Movement.MOVE_RIGHT)
                    self.move_right_arr_timer = curr_timer

        if self.move_down:
            curr_timer = pygame.time.get_ticks()
            if curr_timer - self.move_right_arr_timer > ARR:
                self.grid.move_active_tetromino(Movement.SOFT_DROP)
                self.move_right_arr_timer = curr_timer

        self.grid.apply_gravity()
        self.grid.apply_locking()

    def draw(self):
        self.screen.fill((0, 0, 0))
        self.grid.draw(self.screen)
        pygame.display.update()


if __name__ == '__main__':
    game_controller = GameController()
    pygame.init()

    while True:
        game_controller.handle_events()
        game_controller.draw()