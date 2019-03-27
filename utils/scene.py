from utils.shapes import Sphere, Cuboid, Tetrahedron
from utils.renderer import Renderer, Tri, Lit, Cam
from random import randint, uniform, shuffle
import cv2
import os
import numpy as np


class Scene:

    def __init__(self, gridlines_on=None, gridlines_width=None, gridlines_spacing=None):
        if gridlines_on or gridlines_width or gridlines_spacing:
            assert not (gridlines_on is None\
                or gridlines_width is None\
                or gridlines_spacing is None),\
                "All gridlines variables must be set if any are"

        self.rend = Renderer()

        self.shapes = []
        self.grid_shapes = []

        self.center = (0, 140, 300)

        self.background_prims = []
        background_lower_bound = -1e3
        background_upper_bound = 1e3
        wall_bound = 1e3
        self.background_prims.append(
            Tri([(-wall_bound, 0, wall_bound),
                (wall_bound, 0, wall_bound),
                (-wall_bound, 0, -wall_bound)]))
        self.background_prims.append(
            Tri([(-wall_bound, 0, -wall_bound),
                (wall_bound, 0, wall_bound),
                (wall_bound, 0, -wall_bound)]))
        self.background_prims.append(
            Tri([(-wall_bound, -50, wall_bound),
                (0, wall_bound, wall_bound),
                (wall_bound, -50, wall_bound)]))

        if gridlines_on:
            for i in range(int((background_upper_bound - background_lower_bound) / (gridlines_width + gridlines_spacing))):
                offset = i * (gridlines_width + gridlines_spacing)
                self.grid_shapes.append(Tri([(background_lower_bound + offset, 0.01, background_lower_bound),
                                            (background_lower_bound + offset, 0.01, background_upper_bound),
                                            (background_lower_bound + gridlines_width + offset, 0.01, background_lower_bound)]))
                self.grid_shapes.append(Tri([(background_lower_bound + offset, 0.01, background_upper_bound),
                                            (background_lower_bound + gridlines_width + offset, 0.01, background_upper_bound),
                                            (background_lower_bound + gridlines_width + offset, 0.01, background_lower_bound)]))
                self.grid_shapes.append(Tri([(background_lower_bound, 0.01, background_lower_bound + gridlines_width + offset),
                                             (background_upper_bound, 0.01, background_lower_bound + offset),
                                             (background_lower_bound, 0.01, background_lower_bound + offset)]))
                self.grid_shapes.append(Tri([(background_upper_bound, 0.01, background_lower_bound + offset),
                                             (background_lower_bound, 0.01, background_lower_bound + gridlines_width + offset),
                                            (background_upper_bound, 0.01, background_lower_bound + gridlines_width + offset)]))

        self.light = Lit((400, 300, -800), 1000000)
        self.camera = Cam((0, 80, 140), (128, 128))


    def add_object(self):
        shape = [Sphere(self.center, 0.5), Tetrahedron(self.center), Cuboid(self.center)][randint(0,2)]
        shape.scale(randint(25,40))
        self.__rotate_object(shape)
        self.__translate_object(shape)
        self.shapes.append(shape)

    def crossover(self, scene):
        offspring = Scene()
        offspring.shapes = self.shapes + scene.shapes
        shuffle(offspring.shapes)
        offspring.shapes = offspring.shapes[:len(offspring.shapes)//2]
        return offspring

    def mutate(self):
        if randint(0,1) == 0:
            self.add_object()
        else:
            shape = self.shapes[randint(0, len(self.shapes) - 1)]
            mutation = [self.__scale_object, self.__translate_object, self.__rotate_object][randint(0,2)]
            mutation(shape)

    def __scale_object(self, shape):
        for i in range(3):
            shape.scale(uniform(0.3, 1.7), axis=i)

    def __translate_object(self, shape):
        lowest_y = 1e6
        if type(shape) == Sphere:
            lowest_y = shape.center[1] - shape.radius
        else:
            for tri in shape.render():
                for tup in tri.data:
                    lowest_y = min(lowest_y,tup[1])
        shape.translate((0, -lowest_y, 0))

    def __rotate_object(self, shape):
        shape.rotate(randint(0, 359), randint(0, 359))

    def render(self):
        surface_prims = []
        for shape in self.shapes:
            surface_prims += shape.render()
        views = [self.camera.view_from(0, -.3, 1)]
        res_x, res_y = self.camera.resolution
        return self.rend.render(views, self.light, surface_prims, self.background_prims, res_x, res_y, self.grid_shapes, grid_color=(0.7,0.7,0.7))

if __name__ == '__main__':
    g = Scene(True, gridlines_width=20, gridlines_spacing=30)
    g.add_object()
    shadows, noshadows = g.render()
    # shadows = cv2.cvtColor(shadows.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    # noshadows = cv2.cvtColor(noshadows.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    if not os.path.isdir("tmp_scenes"):
        os.mkdir("tmp_scenes")
    print(shadows.shape)

    cv2.imwrite(os.path.join("tmp_scenes","shadows.png"), shadows)
    cv2.imwrite(os.path.join("tmp_scenes","noshadows.png"), noshadows)
