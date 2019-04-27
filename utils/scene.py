from utils.shapes import Sphere, Cuboid, Tetrahedron, Torus
from utils.renderer import Renderer, Tri, Lit, Cam
from utils.helpers import mean
from random import randint, uniform, shuffle
import cv2
import os
import numpy as np
import math
from numpy.linalg import norm

class Scene:

    def __init__(self, light_variability=(20,8), gridlines_on=None, gridlines_width=None, gridlines_spacing=None):
        if gridlines_on or gridlines_width or gridlines_spacing:
            assert not (gridlines_on is None\
                or gridlines_width is None\
                or gridlines_spacing is None),\
                "All gridlines variables must be set if any are"

        self.rend = Renderer()

        self.shapes = []
        self.grid_shapes = []

        self.center = np.array((0, 140, 300))

        self.light_variability = light_variability

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

        self.default_light = np.array((400, 300, -800))
        self.default_intensity = 1000000
        self.camera = Cam((0, 140, 300), (128, 128))


    def calc_center(self):
        return mean([shape.center for shape in self.shapes])

    def add_object(self, i=randint(0,3)):
        shape = [Tetrahedron(self.center), Cuboid(self.center), Torus(self.center, 0.5, 50, 0.2), Sphere(self.center, 0.5)][i]
        shape.scale(35)
        self.shapes.append(shape)

    def mutate_object(self, shape):
        shape.scale(randint(25, 40))
        self.__rotate_object(shape)
        self.__translate_object(shape)

    def mutate_all_objects(self):
        for shape in self.shapes:
            # self.__scale_object(shape)
            # self.__rotate_object(shape)
            self.__translate_object(shape)

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


    def new_light(self, theta = 60, phi=8):
        d = norm(self.default_light - self.center)
        x_trans = d * math.sin(math.radians(theta))
        z_trans = d * math.cos(math.radians(theta))
        y_trans = d * math.sin(math.radians(phi))
        translation = np.array((x_trans, y_trans, -z_trans))
        return Lit(self.center + translation, self.default_intensity)


    def __scale_object(self, shape):
        for i in range(3):
            shape.scale(uniform(0.8, 1.2), axis=i)

    def __translate_object(self, shape):
        shape.translate((randint(-50, 50), 0, randint(-50, 50)))

    def ground_mesh(self):
        for shape in self.shapes:
            lowest_y = shape.lowest_y()
            shape.translate((0, -lowest_y, 0))


    def __rotate_object(self, shape):
        shape.rotate(randint(0, 359), randint(0, 359), randint(0, 359))

    def refocus_camera(self):
        self.camera.location = self.calc_center()

    def render(self):
        surface_prims = []
        light = self.new_light(*self.light_variability)#Lit(self.default_light, self.default_intensity)#
        for shape in self.shapes:
            surface_prims += shape.render()
        views = [self.camera.view_from(-30, 0, 200)]
        res_x, res_y = self.camera.resolution
        return self.rend.render(views, light, surface_prims, self.background_prims, res_x, res_y, self.grid_shapes, grid_color=(0.7,0.7,0.7))

if __name__ == '__main__':
    g = Scene(10, True, gridlines_width=20, gridlines_spacing=30)
    g.add_object()
    g.add_object()
    g.mutate_all_objects()
    g.ground_mesh()
    shadows, noshadows = g.render()
    # shadows = cv2.cvtColor(shadows.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    # noshadows = cv2.cvtColor(noshadows.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    if not os.path.isdir("tmp_scenes"):
        os.mkdir("tmp_scenes")
    print(shadows.shape)

    cv2.imwrite(os.path.join("tmp_scenes","shadows.png"), shadows)
    cv2.imwrite(os.path.join("tmp_scenes","noshadows.png"), noshadows)
