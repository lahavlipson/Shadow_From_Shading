from renderer import Cir, Tri
import numpy as np

class Shape:
    def __init__(self, center):
        self.center = np.array(center)
        self.rotation_matrix = np.identity(3)

    def translate(self, offset):
        self.center = self.center + offset

    def rotate(self, pitch, yaw):
        c = np.cos(np.deg2rad(pitch))
        s = np.sin(np.deg2rad(pitch))
        pitch_matrix = np.array(((c, -s, 0),
                                 (s,  c, 0),
                                 (0,  0, 1)))
        self.rotation_matrix = np.dot(pitch_matrix, self.rotation_matrix)

        c = np.cos(np.deg2rad(yaw))
        s = np.sin(np.deg2rad(yaw))
        yaw_matrix = np.array(((1, 0,  0),
                               (0, c, -s),
                               (0, s,  c)))
        self.rotation_matrix = np.dot(yaw_matrix, self.rotation_matrix)

    def scale(self, factor, axis=None):
        if axis is None:
            self.scale_matrix = factor * self.scale_matrix
            return

        assert axis >= 0 and axis < 3
        self.scale_matrix[axis] = self.scale_matrix[axis] * factor

    def render(self):
        dimensions = self.triangle_faces.shape
        modified = self.triangle_faces.reshape(dimensions[0] * dimensions[1], 3)
        modified = np.dot(modified, self.scale_matrix)
        modified = np.dot(modified, self.rotation_matrix)
        modified = modified.reshape(dimensions[0], dimensions[1], 3)
        offset = np.tile(self.center, (dimensions[0], dimensions[1], 1))
        modified += offset
        return [Tri(tuple(face)) for face in modified]

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

                               ((-0.5, -0.5, -0.5), (-0.5,  0.5, -0.5), ( 0.5,  0.5, -0.5)),
                               ((-0.5, -0.5, -0.5), ( 0.5,  0.5, -0.5), ( 0.5, -0.5, -0.5)),
                               ((-0.5, -0.5, -0.5), (-0.5,  0.5,  0.5), (-0.5,  0.5, -0.5)),
                               ((-0.5, -0.5, -0.5), (-0.5, -0.5,  0.5), (-0.5,  0.5,  0.5)),
                               ((-0.5, -0.5, -0.5), ( 0.5, -0.5, -0.5), ( 0.5, -0.5,  0.5)),
                               ((-0.5, -0.5, -0.5), ( 0.5, -0.5,  0.5), (-0.5, -0.5,  0.5))]
        self.triangle_faces = np.array(self.triangle_faces)

class Tetrahedron(Shape):
    def __init__(self, center):
        super().__init__(center)
        self.scale_matrix = np.identity(3)
        # Points taken from here: https://en.wikipedia.org/wiki/Tetrahedron
        points = [( 1,  0, -1/np.sqrt([2])),
                  (-1,  0, -1/np.sqrt([2])),
                  ( 0,  1,  1/np.sqrt([2])),
                  ( 0, -1,  1/np.sqrt([2]))]

        self.triangle_faces = [(points[0], points[1], points[2]),
                               (points[0], points[2], points[3]),
                               (points[0], points[3], points[1]),
                               (points[3], points[2], points[1])]
        self.triangle_faces = np.array(self.triangle_faces)


if __name__ == '__main__':
    sp = Sphere((0, 0, 0), 5)
    shapes_so_far = sp.render()
    cuboid = Cuboid((0, 0, 0), 1, 2, 3)
    print(cuboid.render())
