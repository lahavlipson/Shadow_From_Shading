from shapes import Sphere, Cuboid, Tetrahedron
from renderer import Renderer, Cam, Lit, Tri
from random import randint, uniform



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
        self.camera = Cam((0, 382*0.6, -273*0.6), (0, -.3, 1), (640, 480))


    def add_object(self):
        shape = [Sphere(self.center, 1), Tetrahedron(self.center), Cuboid(self.center)][randint(0,2)]
        for i in range(3):
            shape.scale(randint(15,40), axis=i)
        self.__rotate_object(shape)
        self.__translate_object(shape)
        self.shapes.append(shape)

    def mutate(self):
        shape = self.shapes[randint(0, len(self.shapes) - 1)]
        mutation = [self.__scale_object, self.__translate_object, self.__rotate_object][randint(0,2)]
        print("\n\n\n", shape, mutation,"\n\n\n")
        mutation(shape)

    def __scale_object(self, shape):
        for i in range(3):
            shape.scale(uniform(0.3, 1.7), axis=i)

    def __translate_object(self, shape):
        shape.translate((randint(-30,30), randint(-30,30), randint(-30,30)))

    def __rotate_object(self, shape):
        shape.rotate(randint(0, 359), randint(0, 359))

    def render(self):
        surface_prims = []
        for shape in self.shapes:
            surface_prims += shape.render()
        return self.rend.render(self.camera, self.light, surface_prims, self.background_prims)


g = Scene()
for _ in range(3):
    g.add_object()
shadow_numpy_arr, shadowless_numpy_arr = g.render()
g.mutate()
shadow_numpy_arr, shadowless_numpy_arr = g.render()