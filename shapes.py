class Shape:

    def __init__(self, center):
        self.center = center

    def translate(self, offset):
        self.center = self.center + offset


class Sphere(Shape):
    def __init__(self, center, radius):
        super().__init__(center)
        self.radius = radius

if __name__ == '__main__':
    sp = Sphere((0, 0, 0), 5)
