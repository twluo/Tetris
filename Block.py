import collections

import pygame
import sys
import random
from enum import Enum
import numpy as np
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers.core import Dense, Dropout
from random import randint

# RENDERING CONSTANTS
SCALING_FACTOR = 2
WINDOW_HEIGHT = 500 * SCALING_FACTOR
WINDOW_WIDTH = 1000 * SCALING_FACTOR
BLOCK_HEIGHT = 16 * SCALING_FACTOR
BLOCK_WIDTH = 16 * SCALING_FACTOR

# GAME CONSTANTS
GRID_WIDTH = 10
GRID_HEIGHT = 40
MAX_HEIGHT = 20
STARTING_HEIGHT = GRID_HEIGHT - MAX_HEIGHT
CELL_NUM = GRID_WIDTH * GRID_HEIGHT
TETROMINO_BOX_WIDTH = 5
TETROMINO_BOX_HEIGHT = 5

# GAMEPLAY CONSTANTS
GRAVITY_DELAY = 1000
LOCK_DELAY = 1000
DAS = 100
ARR = 30
NEXT_LENGTH = 5

# AI CONSTANTS
RATE_OF_ACTION = 0


def check_bounds(x, y):
    return 0 <= x <= GRID_HEIGHT and 0 <= y <= GRID_WIDTH


def spawn_bag():
    seven_bag = [k for k in Tetromino]
    random.shuffle(seven_bag)
    return seven_bag


def list_moves():
    return [k for k in Movement]


def clamp(value, min_value=0, max_value=GRID_WIDTH - 1):
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

def flatten_grid(grid):
    return np.array(grid).flatten()


class Movement(Enum):
    MOVE_LEFT = 0
    MOVE_RIGHT = 1
    SOFT_DROP = 2
    HARD_DROP = 3
    ROTATE_LEFT = 4
    ROTATE_RIGHT = 5
    HOLD = 6


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
    I_BLOCK = (0, 255, 255)
    L_BLOCK = (255, 165, 0)
    J_BLOCK = (0, 0, 255)
    O_BLOCK = (255, 255, 0)
    S_BLOCK = (0, 255, 0)
    T_BLOCK = (128, 0, 128)
    Z_BLOCK = (255, 0, 0)


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
        new_color = [x for x in new_color]
        self.image.fill(new_color, None, pygame.BLEND_RGBA_MULT)


class BasicTetromino(object):

    def __init__(self, tetromino_id, position_offset=(0, -3)):
        self.tetromino_id = tetromino_id
        self.rotation = 0
        self.configurations = [[(0, 0) for x in range(TETROMINO_BOX_WIDTH) for x in range(TETROMINO_BOX_HEIGHT)]]
        self.position = (4 - position_offset[0], STARTING_HEIGHT + position_offset[1])
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
            if not self.check_collision(new_position, new_rot, grid):
                return True, new_position
        return False, self.position

    def move_right(self, grid):
        position = (clamp(self.position[0] + 1), self.position[1])
        if not self.check_collision(position, self.rotation, grid):
            self.position = position
            return True
        return False

    def move_left(self, grid):
        position = (clamp(self.position[0] - 1), self.position[1])
        if not self.check_collision(position, self.rotation, grid):
            self.position = position
            return True
        return False

    def soft_drop(self, grid):
        position = self.position[0], self.position[1] + 1
        if not self.check_collision(position, self.rotation, grid):
            self.position = position
            return True
        return False

    def hard_drop(self, grid):
        for y_position in range(self.position[1], GRID_HEIGHT + 1):
            curr_position = (self.position[0], y_position)
            next_position = (self.position[0], y_position + 1)
            if self.check_collision(next_position, self.rotation, grid):
                self.position = curr_position
                return True
        return False

    def check_collision(self, position, rotation, grid):
        for x_offset, y_offset in self.configurations[rotation]:
            x = position[0] + x_offset
            y = position[1] + y_offset
            if y < GRID_HEIGHT and 0 <= x < GRID_WIDTH:
                if grid[y][x] is not 0 and x >= 0:
                    return True
            else:
                return True
        else:
            return False

    def is_game_over(self, grid):
        return self.check_collision(self.position, self.rotation, grid)

    def __str__(self):
        return str(self.tetromino_id)

    def __repr__(self):
        return str(self.tetromino_id)


class ITetromino(BasicTetromino):

    def __init__(self):
        super(ITetromino, self).__init__(Tetromino.I_BLOCK, (0, -2))
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
    def __init__(self, name, grid_center):
        self.name = name
        self.buffer = WINDOW_WIDTH/50
        self.grid = [[0 for x in range(GRID_WIDTH)] for x in range(GRID_HEIGHT)]
        self.gridBW = [[0 for x in range(GRID_WIDTH)] for x in range(GRID_HEIGHT)]
        self.grid_size = (GRID_WIDTH * BLOCK_WIDTH, MAX_HEIGHT * BLOCK_HEIGHT)
        self.grid_center = grid_center
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
        self.score_top_left = (self.hold_top_left[0], self.hold_top_left[1] + self.hold_size[1]/2 + self.buffer)
        self.seven_bag = spawn_bag()
        self.next_bag = spawn_bag()
        self.active_tetromino = None
        self.next_tetrominos = [get_tetromino(self.seven_bag.pop(0)) for x in range(NEXT_LENGTH)]
        self.gravity_timer = 0
        self.lock_timer = 0
        self.game_over = False
        self.hold_flag = False
        self.hold_tetromino = None
        self.score = 0
        self.combo = 0
        self.startTime = pygame.time.get_ticks()
        self.counting_time = 0

    def pretty_print(self):
        for x in self.grid:
            print(x.value) if x in Tetromino else print(x)

    def swap_tetromino(self):
        if self.hold_flag is True:
            return self.gridBW, 0, self.game_over
        self.hold_flag = True
        self.clear_active_tetromino()
        if self.hold_tetromino is None:
            self.hold_tetromino = get_tetromino(self.active_tetromino.tetromino_id)
            self.spawn_new_tetromino()
            return self.gridBW, 0, self.game_over
        new_tetromino = self.hold_tetromino.tetromino_id
        self.hold_tetromino = get_tetromino(self.active_tetromino.tetromino_id)
        self.spawn_tetromino(new_tetromino)
        return self.gridBW, 0, self.game_over

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
        if self.active_tetromino.is_game_over(self.grid):
            print("Game Over")
            self.game_over = True
        self.update_grid()

    def apply_gravity(self):
        curr_time = pygame.time.get_ticks()
        if curr_time - self.gravity_timer >= GRAVITY_DELAY:
            self.move_active_tetromino(Movement.SOFT_DROP)
            self.gravity_timer = curr_time

    def apply_locking(self):
        curr_time = pygame.time.get_ticks()
        if curr_time - self.lock_timer >= LOCK_DELAY:
            self.move_active_tetromino(Movement.HARD_DROP)

    def update_grid(self):
        for x_offset, y_offset in self.active_tetromino.configurations[self.active_tetromino.rotation]:
            x = self.active_tetromino.position[1] + y_offset
            y = self.active_tetromino.position[0] + x_offset
            if check_bounds(x, y):
                self.grid[x][y] = self.active_tetromino.tetromino_id
                self.gridBW[x][y] = 1
        # self.pretty_print()

    def clear_active_tetromino(self):
        for x_offset, y_offset in self.active_tetromino.configurations[self.active_tetromino.rotation]:
            x = self.active_tetromino.position[1] + y_offset
            y = self.active_tetromino.position[0] + x_offset
            if check_bounds(x, y):
                self.grid[x][y] = 0
                self.gridBW[x][y] = 0

    def move_active_tetromino(self, movement):
        if movement is Movement.HOLD:
            return self.swap_tetromino()
        self.clear_active_tetromino()
        has_it_moved = self.active_tetromino.move(movement, self.grid)
        self.update_grid()
        reward = -0.01
        if has_it_moved:
            self.gravity_timer = pygame.time.get_ticks()
            self.lock_timer = pygame.time.get_ticks()
        if movement is Movement.HARD_DROP:
            reward = self.check_and_clear_lines() * 2
            self.spawn_new_tetromino()
        return self.gridBW, reward, self.game_over

    def check_and_clear_lines(self):
        counter = 0
        for row in self.grid:
            for block in row:
                if block is 0:
                    break
            else:
                self.grid.remove(row)
                self.grid.insert(STARTING_HEIGHT, [0 for x in range(GRID_WIDTH)])
                counter += 1
        self.combo = 0 if counter is 0 else self.combo + counter
        self.score += counter
        return counter

    def draw(self, screen, game_over):

        font = pygame.font.SysFont("comicsansms", 20)
        if not game_over:
            self.counting_time = pygame.time.get_ticks() - self.startTime
        else:
            if not self.game_over:
                text = font.render("Congratulation", True, (0, 128, 0))
            else:
                text = font.render("Better Luck Next Time", True, (0, 128, 0))
            screen.blit(text,
                        (self.score_top_left[0], self.score_top_left[1] + 4 * text.get_height()))
        text = font.render("Score: " + str(self.score), True, (0, 128, 0))
        screen.blit(text,
                    (self.score_top_left[0], self.score_top_left[1] + text.get_height()))
        text = font.render("Combo: " + str(self.combo), True, (0, 128, 0))
        screen.blit(text,
                    (self.score_top_left[0], self.score_top_left[1] + 2 * text.get_height()))
        counting_minutes = str(int(self.counting_time / 60000)).zfill(2)
        counting_seconds = str(int((self.counting_time % 60000) / 1000)).zfill(2)
        counting_millisecond = str(int(self.counting_time % 1000)).zfill(3)
        counting_string = "%s:%s:%s" % (counting_minutes, counting_seconds, counting_millisecond)
        text = font.render("Timer: " + counting_string, True, (0, 128, 0))
        screen.blit(text,
                    (self.score_top_left[0], self.score_top_left[1] + 3 * text.get_height()))
        pygame.draw.rect(screen, (255, 255, 255), (self.grid_top_left, self.grid_size), 1)
        for x in range(GRID_WIDTH):
            for y in range(STARTING_HEIGHT - 3, GRID_HEIGHT):
                if self.grid[y][x] is not 0:
                    block = Block(Tetromino(self.grid[y][x]))
                    origin_x = self.grid_center[0] - (self.grid_size[0] - BLOCK_HEIGHT * (2 * x + 1)) / 2
                    origin_y = self.grid_center[1] - (self.grid_size[1] - BLOCK_WIDTH * (2*(y - STARTING_HEIGHT) + 1))/2
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


class AIController:
    def __init__(self, move_list):
        self.move_list = move_list
        self.alpha = 0.1
        self.gamma = 0.9
        self.epsilon = 0.5
        self.memory = collections.deque(maxlen=2500)
        self.model = self.network()
        self.weight_path = 'weights.hdf5'
        try:
            self.model.load_weights(self.weight_path)
        except Exception as e:
            print("No File")
        # self.q_table = np.zeroes([1, len(self.move_list)])

    def network(self):
        model = Sequential()
        model.add(Dense(input_dim=CELL_NUM, output_dim=len(self.move_list), activation='softmax'))
        opt = Adam(self.alpha)
        model.compile(loss='mse', optimizer=opt)
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay_new(self, memory, batch_size):
        if len(memory) > batch_size:
            minibatch = random.sample(memory, batch_size)
        else:
            minibatch = memory
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(np.array([next_state]))[0])
            target_f = self.model.predict(np.array([state]))
            target_f[0][np.argmax(action)] = target
            self.model.fit(np.array([state]), target_f, epochs=1, verbose=0)

    def train_short_memory(self, state, action, reward, next_state, done):
        target = reward
        if not done:
            target = reward + self.gamma * np.amax(self.model.predict(next_state.reshape((1, CELL_NUM)))[0])
        target_f = self.model.predict(state.reshape((1, CELL_NUM)))
        target_f[0][np.argmax(action)] = target
        self.model.fit(state.reshape((1, CELL_NUM)), target_f, epochs=1, verbose=0)

    def pick_a_move(self):
        random.shuffle(self.move_list)
        return self.move_list[0]


class GameController(object):
    def __init__(self, ai_controller):
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        self.player = Grid("player", (WINDOW_WIDTH/4, WINDOW_HEIGHT/2))
        self.player.spawn_new_tetromino()
        self.player.update_grid()
        self.ai = Grid("ai", (3*WINDOW_WIDTH/4, WINDOW_HEIGHT/2))
        self.ai.spawn_new_tetromino()
        self.ai.update_grid()
        self.move_left = False
        self.move_right = False
        self.move_down = False
        self.move_list = list_moves()
        self.move_left_das_timer = pygame.time.get_ticks()
        self.move_left_arr_timer = pygame.time.get_ticks()
        self.move_right_das_timer = pygame.time.get_ticks()
        self.move_right_arr_timer = pygame.time.get_ticks()
        self.move_down_arr_timer = pygame.time.get_ticks()
        self.ai_movement_timer = pygame.time.get_ticks()
        self.is_game_over = False, False
        self.ai_controller = ai_controller
        self.game_count = 0
        self.startTime = pygame.time.get_ticks()

    def exit(self):
        self.ai_controller.model.save_weights(self.ai_controller.weight_path)
        pygame.quit()
        sys.exit(0)

    def handle_player_inputs(self):
        event = pygame.event.poll()
        old_grid = flatten_grid(self.player.gridBW)
        if self.player.game_over:
            self.is_game_over = True, False
        if event.type == pygame.QUIT:
            self.exit()
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_a:
                self.move_left = True
                self.move_right = False
                self.player.move_active_tetromino(Movement.MOVE_LEFT)
            if event.key == pygame.K_d:
                self.move_right = True
                self.move_left = False
                self.player.move_active_tetromino(Movement.MOVE_RIGHT)
            if event.key == pygame.K_s:
                self.move_down = True
                self.player.move_active_tetromino(Movement.SOFT_DROP)
            if event.key == pygame.K_COMMA:
                self.player.move_active_tetromino(Movement.ROTATE_LEFT)
            if event.key == pygame.K_PERIOD:
                self.player.move_active_tetromino(Movement.ROTATE_RIGHT)
            if event.key == pygame.K_SPACE:
                self.player.move_active_tetromino(Movement.HOLD)
            if event.key == pygame.K_w:
                self.player.move_active_tetromino(Movement.HARD_DROP)
            if event.key == pygame.K_r:
                self.reset()

        keys = pygame.key.get_pressed()
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
                    self.player.move_active_tetromino(Movement.MOVE_LEFT)
                    self.move_left_arr_timer = curr_timer

        if self.move_right:
            curr_timer = pygame.time.get_ticks()
            if curr_timer - self.move_right_das_timer > DAS:
                if curr_timer - self.move_right_arr_timer > ARR:
                    self.player.move_active_tetromino(Movement.MOVE_RIGHT)
                    self.move_right_arr_timer = curr_timer

        if self.move_down:
            curr_timer = pygame.time.get_ticks()
            if curr_timer - self.move_right_arr_timer > ARR:
                self.player.move_active_tetromino(Movement.SOFT_DROP)
                self.move_right_arr_timer = curr_timer

        self.player.apply_gravity()
        self.player.apply_locking()

    def handle_ai_inputs(self):
        curr_timer = pygame.time.get_ticks()
        if self.ai.game_over:
            self.is_game_over = False, True
        if curr_timer - self.ai_movement_timer > RATE_OF_ACTION:
            old_grid = flatten_grid(self.ai.gridBW)
            epsilon = 1 - (self.game_count * 1/500)
            if randint(0, 10) < epsilon * 10:
                ai_move = self.ai_controller.pick_a_move()
                print("randomly picked " + str(ai_move))
            else:
                prediction = self.ai_controller.model.predict(old_grid.reshape(1, CELL_NUM))
                ai_move = Movement(np.argmax(prediction[0]))
                print("intelligently picked " + str(ai_move))
            new_grid, reward, is_game_over = self.ai.move_active_tetromino(ai_move)
            new_grid_flat = flatten_grid(new_grid)
            if self.ai.game_over:
                print("Lost better try again")
                reward = -10
            self.ai_controller.train_short_memory(old_grid, ai_move, reward, new_grid_flat, is_game_over)
            self.ai_controller.remember(old_grid, ai_move, reward, new_grid_flat, is_game_over)
            self.ai_movement_timer = curr_timer
        self.ai.apply_gravity()
        self.ai.apply_locking()

    def handle_game_over(self):
        if self.ai.game_over:
            self.game_count += 1
            self.reset()
            print("Next Game: " + str(self.game_count))
        event = pygame.event.poll()
        if event.type == pygame.QUIT:
            self.exit()
        if event.type == pygame.KEYDOWN:
            self.reset()

    def reset(self):
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        self.player = Grid("player", (WINDOW_WIDTH/4, WINDOW_HEIGHT/2))
        self.player.spawn_new_tetromino()
        self.player.update_grid()
        self.ai = Grid("ai", (3*WINDOW_WIDTH/4, WINDOW_HEIGHT/2))
        self.ai.spawn_new_tetromino()
        self.ai.update_grid()
        self.move_left = False
        self.move_right = False
        self.move_down = False
        self.move_list = list_moves()
        self.move_left_das_timer = pygame.time.get_ticks()
        self.move_left_arr_timer = pygame.time.get_ticks()
        self.move_right_das_timer = pygame.time.get_ticks()
        self.move_right_arr_timer = pygame.time.get_ticks()
        self.move_down_arr_timer = pygame.time.get_ticks()
        self.ai_movement_timer = pygame.time.get_ticks()
        self.is_game_over = False, False

    def draw(self, game_over):
        self.screen.fill((0, 0, 0))
        self.player.draw(self.screen, game_over)
        self.ai.draw(self.screen, game_over)
        font = pygame.font.SysFont("comicsansms", 20)
        text = font.render("Game: " + str(self.game_count), True, (0, 128, 0))
        self.screen.blit(text,(400, 500))
        font = pygame.font.SysFont("comicsansms", 20)
        counting_time = pygame.time.get_ticks() - self.startTime
        counting_minutes = str(int(counting_time / 60000)).zfill(2)
        counting_seconds = str(int((counting_time % 60000) / 1000)).zfill(2)
        counting_millisecond = str(int(counting_time % 1000)).zfill(3)
        counting_string = "%s:%s:%s" % (counting_minutes, counting_seconds, counting_millisecond)
        text = font.render("Timer: " + counting_string, True, (0, 128, 0))
        self.screen.blit(text,(400, 520))
        pygame.display.update()


if __name__ == '__main__':
    game_controller = GameController(AIController(list_moves()))
    pygame.init()

    while True:
        game_over = game_controller.is_game_over[0] or game_controller.is_game_over[1]
        if not game_over:
            game_controller.handle_player_inputs()
            game_controller.handle_ai_inputs()
        else:
            game_controller.handle_game_over()
        game_controller.draw(game_over)
