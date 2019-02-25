from renderer import Cir, Tri
import numpy as np

class Shape:
    def __init__(self, center):
        self.center = center

    def translate(self, offset):
        self.center = self.center + offset

    def rotate(self, pitch, yaw):
        pass

    def render(self):
        pass

    def scale(self, factor, axis=None):
       pass

class Sphere(Shape):
    def __init__(self, center, radius):
        super().__init__(center)
        self.radius = radius

    def scale(self, factor, axis=None):
        self.radius = radius * factor

    def render(self):
        # Returns a list for sake of consistency with other render methods
        return [Cir((self.center, self.radius))]

class Cuboid(Shape):
    def __init__(self, center):
        super().__init__(center)
        self.scale_matrix = np.identity(3)
        self.triangle_faces = [(( 0.5,  0.5,  0.5), (-0.5,  0.5, -0.5), (-0.5,  0.5,  0.5)),
                               (( 0.5,  0.5,  0.5), (-0.5,  0.5,  0.5), (-0.5, -0.5,  0.5)),
                               (( 0.5,  0.5,  0.5), ( 0.5, -0.5,  0.5), ( 0.5, -0.5, -0.5)),
                               (( 0.5,  0.5,  0.5), (-0.5, -0.5,  0.5), ( 0.5, -0.5,  0.5)),
                               (( 0.5,  0.5,  0.5), ( 0.5, -0.5, -0.5), ( 0.5,  0.5, -0.5)),
                               (( 0.5,  0.5,  0.5), ( 0.5,  0.5, -0.5), (-0.5,  0.5, -0.5)),

                               ((-0.5, -0.5, -0.5), (-0.5, -0.5, -0.5), (-0.5, -0.5, -0.5)),
                               ((-0.5, -0.5, -0.5), (-0.5, -0.5, -0.5), (-0.5, -0.5, -0.5)),
                               ((-0.5, -0.5, -0.5), (-0.5, -0.5, -0.5), (-0.5, -0.5, -0.5)),
                               ((-0.5, -0.5, -0.5), (-0.5, -0.5, -0.5), (-0.5, -0.5, -0.5)),
                               ((-0.5, -0.5, -0.5), (-0.5, -0.5, -0.5), (-0.5, -0.5, -0.5)),
                               ((-0.5, -0.5, -0.5), (-0.5, -0.5, -0.5), (-0.5, -0.5, -0.5))]
        self.triangle_faces = np.array(self.triangle_faces)


    def scale(self, factor, axis=None):
        if axis is None:
            self.scale_matrix = factor * self.scale_matrix
            return

        assert axis >= 0 and axis < 3
        self.scale_matrix[axis] = self.scale_matrix[axis] * factor

    def render(self):
        modified = self.triangle_faces.reshape(36, 3)
        modified = np.dot(modified, self.scale_matrix)
        modified = modified.reshape(12, 3, 3)
        return [Tri(tuple(face)) for face in modified]


if __name__ == '__main__':
    sp = Sphere((0, 0, 0), 5)
    shapes_so_far = sp.render()
    cuboid = Cuboid((0, 0, 0), 1, 2, 3)
    print(cuboid.render())
