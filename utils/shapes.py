from utils.renderer import Cir, Tri
import numpy as np
from random import randint, uniform, shuffle
from math import sqrt, pi


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

    def rotate(self, pitch, yaw):
        pass

    def scale(self, factor, axis=None):
        if axis == 0 or axis is None:
            self.radius *= factor

    def render(self):
        # Returns a list for sake of consistency with other render methods
        return [Cir((self.center, self.radius))]

    def __str__(self):
        return "Sphere"

class Torus(Shape):
    def __init__(self, center, radius, num_spheres, wall_radius):
        super().__init__(center)
        self.radius = radius
        self.wall_radius = wall_radius
        self.num_spheres = num_spheres

    def scale(self, factor, axis=None):
        if axis == 0 or axis is None:
            self.radius *= factor
            self.wall_diameter *= factor

    def render(self):
        sphere_centers = []
        inter_sphere_angle = 2 * pi / self.num_spheres

        for i in range(self.num_spheres):
            theta = i * inter_sphere_angle
            sphere_centers.append((np.sin(theta), np.cos(theta), 0))

        print(sphere_centers)
        return []


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

    def __str__(self):
        return "Cuboid"

class Tetrahedron(Shape):
    def __init__(self, center):
        super().__init__(center)
        self.scale_matrix = np.identity(3)
        # Points taken from here: https://en.wikipedia.org/wiki/Tetrahedron
        points = [( 1,  0, -1/sqrt(2)),
                  (-1,  0, -1/sqrt(2)),
                  ( 0,  1,  1/sqrt(2)),
                  ( 0, -1,  1/sqrt(2))]

        self.triangle_faces = [(points[0], points[1], points[2]),
                               (points[0], points[2], points[3]),
                               (points[0], points[3], points[1]),
                               (points[3], points[2], points[1])]
        self.triangle_faces = np.array(self.triangle_faces)

    #Needs to be tested further
    def expand(self):
        side = self.triangle_faces[randint(0,self.triangle_faces.shape[0]-1)]
        print("expand called!", side.shape, self.triangle_faces.shape)
        centroid = (sum(side[:,0])/3, sum(side[:,1])/3, sum(side[:,2])/3) + 0.6*np.cross(side[2]-side[1],side[0]-side[1])
        f1 = np.array([side[0],side[1],centroid]).reshape((1,3,3))
        f2 = np.array([side[1], side[2], centroid]).reshape((1,3,3))
        f3 = np.array([centroid, side[2], side[0]]).reshape((1,3,3))

        self.triangle_faces = np.concatenate((self.triangle_faces, f1,f2,f3),axis=0)

    def __str__(self):
        return "Tetrahedron"


if __name__ == '__main__':
    sp = Sphere((0, 0, 0), 5)
    shapes_so_far = sp.render()
    cuboid = Cuboid((0, 0, 0), 1, 2, 3)
    print(cuboid.render())
