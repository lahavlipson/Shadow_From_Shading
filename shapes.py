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

    def __init__(self, center, width, height, depth):
        super().__init__(center)
        self.side_lengths = np.array(width, height, depth)

    def scale(self, factor, axis=None):
        assert axis >= 0 and axis < 3
        self.side_lengths[axis] = self.side_lengths[axis] * factor

    def render(self):
        pass

if __name__ == '__main__':
    sp = Sphere((0, 0, 0), 5)
    shapes_so_far = sp.render()
