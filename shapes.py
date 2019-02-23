class Shape:

    def __init__(self, center):
        self.center = center

    def translate(self, offset):
        self.center = self.center + offset
