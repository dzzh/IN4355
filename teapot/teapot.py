import sys
import pygame
import math
import classes
from random import randrange

TEST_FILE = 'tests/teapot.txt'
WINDOW_WIDTH = 700
WINDOW_HEIGHT = 500

def read_file(file):
    """Read 3D-model from the given file and returns it as a list of triangles"""

    triangles = []

    with open(TEST_FILE, 'r') as a_file:
        for a_line in a_file:
            str = a_line.split(",")
            triangle = []
            for word in str:
                number = word.strip("{}()\n\r\"")
                triangle.append(float(number))
            triangles.append(triangle)

    return triangles


def random_color():
    """Return random color in form of (red, green, blue) with 32 values per channel"""
    return tuple([randrange(1, 256, 8) - 1 for _ in range(3)])


def handle_events():
    """Execution loop"""

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit(0)

def model_boundaries():
    """Find model boundaries"""

    min_x = 1
    max_x = -1
    min_y = 1
    max_y = -1

    #Find model boundaries
    for triangle in triangles:
        for index, point in enumerate(triangle):
            if index % 2 == 0:
                if point < min_x:
                    min_x = point
                elif point > max_x:
                    max_x = point
            else:
                if point < min_y:
                    min_y = point
                elif point > max_y:
                    max_y = point

    return classes.model_boundaries(min_x,max_x,min_y,max_y, WINDOW_WIDTH, WINDOW_HEIGHT)

if __name__ == '__main__':
    triangles = read_file(TEST_FILE)

    pygame.init()

    window = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))

    window_center_x = WINDOW_WIDTH / 2
    window_center_y = WINDOW_HEIGHT / 2

    model_boundaries = model_boundaries()

    scale = model_boundaries.scale_factor()
    shift_x = model_boundaries.shift_x()
    shift_y = model_boundaries.shift_y()

    for triangle in triangles:
        new_point = []
        for index, point in enumerate(triangle):
            if index % 2 == 0:
                new_point.append(window_center_x + point * scale - shift_x)
            else:
                new_point.append(window_center_y - point * scale + shift_y)

        pygame.draw.polygon(window, random_color(),
            ((new_point[0], new_point[1]), (new_point[2], new_point[3]), (new_point[4], new_point[5])))

        pygame.display.update()

    handle_events()

