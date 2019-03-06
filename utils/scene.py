from utils.shapes import Sphere, Cuboid, Tetrahedron
from utils.renderer import Renderer, Tri, Lit, Cam
from random import randint, uniform, shuffle
import cv2 as cv
import os


class Scene:

    def __init__(self):
        self.rend = Renderer()

        self.shapes = []
        
        self.center = (0, 140, 150)

        self.background_prims = []
        self.background_prims.append(
            Tri([(-10000, -40, 1000), (10000, -40, 10000), (-10000, -40, -10000)]))
        self.background_prims.append(
            Tri([(-10000, -40, -10000), (10000, -40, 10000), (10000, -40, -10000)]))
        self.background_prims.append(
            Tri([(-10000, -50, 800), (0, 5000, 800), (10000, -50, 800)]))



        self.light = Lit((100, 500, -80), 130000)
        self.cameras = [Cam((300, 230, -200), (-0.8, -.3, 1), (640, 480)), \
                        Cam((0, 230, -200), (0, -.3, 1), (640, 480)), \
                        Cam((-300, 230, -200), (0.8, -.3, 1), (640, 480))]


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
        shape.translate((randint(-30,30), randint(-30,30), randint(-30,30)))

    def __rotate_object(self, shape):
        shape.rotate(randint(0, 359), randint(0, 359))

    def render(self, name=""):
        surface_prims = []
        for shape in self.shapes:
            surface_prims += shape.render()
        return self.rend.render(self.cameras, self.light, surface_prims, self.background_prims,name=name)

if __name__ == '__main__':
    g = Scene()
    g.add_object()
    shadows, noshadows = g.render()
    if not os.path.isdir("tmp_scenes"):
        os.mkdir("tmp_scenes")
    cv.imwrite(os.path.join("tmp_scenes","shadows.png"), shadows)
    cv.imwrite(os.path.join("tmp_scenes","noshadows.png"), noshadows)