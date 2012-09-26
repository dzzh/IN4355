import sys
import pygame
import math
import classes
from random import randrange

TEST_FILE = 'tests/teapot.txt'
WINDOW_WIDTH = 1300
WINDOW_HEIGHT = 800

DELTA = 0.000001

def read_file(file):
    """Read 3D-model from the given file and returns it as a list of triangles"""

    triangles = []

    with open(file, 'r') as a_file:
        for a_line in a_file:
            str = a_line.split(",")
            raw_data = []
            for word in str:
                number = word.strip("{}()\n\r\"")
                raw_data.append(float(number))
            point1 = classes.point(raw_data[0],raw_data[1])
            point2 = classes.point(raw_data[2],raw_data[3])
            point3 = classes.point(raw_data[4],raw_data[5])
            triangles.append((point1, point2, point3))

    return triangles


def random_color():
    """Return random color in form of tuple (red, green, blue) with 32 values per channel"""
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
        for point in triangle:
            x = point.get_x()
            y = point.get_y()
            if x < min_x:
                min_x = x
            elif x > max_x:
                max_x = x

            if y < min_y:
                min_y = y
            elif y > max_y:
                max_y = y

    return classes.model_boundaries(min_x,max_x,min_y,max_y, WINDOW_WIDTH, WINDOW_HEIGHT)

#TODO make it work with arbitrary list
def contains_equal_values(value1, value2, value3):
    """Return True if at least two out of supplied values are equal, false otherwise,
       is needed to detect whether a triangle is placed with horizontal/vertical lines"""

    if math.fabs(value1-value2) < DELTA or math.fabs(value2-value3) < DELTA or math.fabs(value1-value3) < DELTA:
        return True
    return False

def distance(point1, point2):
    """Distance formula to determine length of a line segment between two points"""
    return math.sqrt(math.pow(point2.get_x() - point1.get_x(),2) + math.pow(point2.get_y() - point1.get_y(),2))

def is_triangle_small(triangle):
    """Return True if every side of a given triangle is less than bound"""

    bound = 1

    if ((distance(triangle[0], triangle[1]) < bound) and
        (distance(triangle[0], triangle[2]) < bound)and
        (distance(triangle[1], triangle[2]) < bound)):
        return True

    return False

def get_medium_point(triangle, index):
    min_val = min(triangle[0].get()[index], triangle[1].get()[index], triangle[2].get()[index])
    max_val = max(triangle[0].get()[index], triangle[1].get()[index], triangle[2].get()[index])

    for point in triangle:
        if min_val < point.get()[index] < max_val:
            return point


def line_equation(point1, point2, medium_point, index):
    if index == 1:
        comp_index = 0
    else:
        comp_index = 1

    value = ((point2.get()[index] - point1.get()[index]) /
            (point2.get()[comp_index] - point1.get()[comp_index]) *
            (medium_point.get()[comp_index] - point1.get()[comp_index]) + point1.get()[index])
    comp_value = medium_point.get()[comp_index]

    if index == 0:
        return classes.point(value, comp_value)
    else:
        return classes.point(comp_value, value)

def render_triangle(triangle):
    """Splits the triangles to get right triangles and renders them"""

    contains_horizontal = contains_equal_values(triangle[0].get_y(),triangle[1].get_y(),triangle[2].get_y())
    contains_vertical   = contains_equal_values(triangle[0].get_x(),triangle[1].get_x(),triangle[2].get_x())

    #draw the triangle if it is right and/or small
    if (contains_horizontal and contains_vertical) or is_triangle_small(triangle):
        draw_triangle(triangle)

    #if triangle already has a horizontal, compose two triangles by adding vertical
    elif contains_horizontal:
        medium_point = get_medium_point(triangle, 0)
        point1 = triangle[0] if triangle[0] != medium_point else triangle[1]
        point2 = triangle[2] if triangle[2] != medium_point else triangle[1]

        split_point = line_equation(point1,point2,medium_point,1)
        render_triangle([point1, medium_point, split_point])
        render_triangle([point2, medium_point, split_point])

    #otherwise compose two triangles with a horizontal
    else:
        medium_point = get_medium_point(triangle, 1)
        point1 = triangle[0] if triangle[0] != medium_point else triangle[1]
        point2 = triangle[2] if triangle[2] != medium_point else triangle[1]

        split_point = line_equation(point1,point2,medium_point,0)
        render_triangle([point1, medium_point, split_point])
        render_triangle([point2, medium_point, split_point])

def draw_triangle(triangle):
    """Draws triangle in a scene"""

    pygame.draw.polygon(window, random_color(),
        (triangle[0].get(), triangle[1].get(), triangle[2].get()))
    pygame.display.update()

if __name__ == '__main__':

    #read model
    triangles = read_file(TEST_FILE)

    #init pygame and compute basic model parameters
    pygame.init()
    window = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))

    window_center_x = WINDOW_WIDTH / 2
    window_center_y = WINDOW_HEIGHT / 2
    model_boundaries = model_boundaries()
    scale = model_boundaries.scale_factor()
    shift_x = model_boundaries.shift_x()
    shift_y = model_boundaries.shift_y()

    #scale and render each triangle
    for triangle in triangles:
        new_triangle = []
        for point in triangle:
            new_triangle.append(classes.point(
                window_center_x + point.get_x() * scale - shift_x,
                window_center_y - point.get_y() * scale + shift_y))

        render_triangle(new_triangle)

    #execution loop
    handle_events()

