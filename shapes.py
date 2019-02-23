from renderer import Cir

class Shape:

    def __init__(self, center):
        self.center = center

    def translate(self, offset):
        self.center = self.center + offset

    def rotate(self, pitch, yaw):
        pass

    def render(self):
        pass

class Sphere(Shape):
    def __init__(self, center, radius):
        super().__init__(center)
        self.radius = radius

    def render(self):
        # Returns a list for sake of consistency with other render methods
        return [Cir((self.center, self.radius))]

if __name__ == '__main__':
    sp = Sphere((0, 0, 0), 5)
    shapes_so_far = sp.render()
