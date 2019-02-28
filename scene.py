from shapes import Sphere, Cuboid, Tetrahedron
from renderer import Renderer, Cam, Lit, Tri
from random import randint, uniform, shuffle



class Scene:

    def __init__(self):
        # self.shapes = []

        self.rend = Renderer("./Renderer")

        self.shapes = []
        
        self.center = (0, 180, 150)

        self.background_prims = []
        self.background_prims.append(
            Tri([(-1000, -40, 1000), (1000, -40, 1000), (-1000, -40, -1000)]))
        self.background_prims.append(
            Tri([(-1000, -40, -1000), (1000, -40, 1000), (1000, -40, -1000)]))
        self.background_prims.append(
            Tri([(-2000, -50, 800), (0, 500, 800), (2000, -50, 800)]))



        self.light = Lit((100, 500, -80), 130000)
        self.cameras = [Cam((0, 230, -164), (0, -.3, 1), (640, 480)), \
                        Cam((30, 230, -164), (0, -.3, 1), (640, 480))]


    def add_object(self):
        shape = [Sphere(self.center, 0.5), Tetrahedron(self.center), Cuboid(self.center)][randint(0,2)]
        for i in range(3):
            shape.scale(randint(15,40), axis=i)
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


