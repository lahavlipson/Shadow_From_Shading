import os
import numpy as np
import sys
from math import sin, cos

if sys.platform[:5] == 'linux':
    print("Running on Linux")
    import renderer_lib_linux as renderer_lib
else:
    print("Running on MacOS")
    import renderer_lib_macos as renderer_lib

SCN = ".scn"
PNG = ".png"
NL = "\n"


class Tri:
    def __init__(self, data):
        if len(data) != 3 or len(data[0]) != 3:
            raise ValueError('Triangle\'s constructor accepts: [(x,y,z),(x,y,z),(x,y,z)]')
        self.data = data

    def __str__(self):
        return "t " + " ".join([' '.join(str(e) for e in t) for t in self.data])

class Cir:
    def __init__(self, data):
        if len(data) != 2 or len(data[0]) != 3:
            raise ValueError('Circle\'s constructor accepts: [(x,y,z),r]')
        self.center = data[0]
        self.r = data[1]

    def __str__(self):
        return "s " + ' '.join(str(e) for e in self.center) + " " + str(self.r)

class Cam:
    def __init__(self, loc, res):
        if len(loc) != 3 or len(res) != 2:
            raise ValueError('Camera\'s constructor accepts: (x,y,z), (h,w)')
        self.location = loc
        self.resolution = res

    def view_from(self, yaw, pitch, distance):
        c = np.cos(np.deg2rad(-pitch))
        s = np.sin(np.deg2rad(-pitch))
        pitch_matrix = np.array(((c, -s, 0),
                                 (s,  c, 0),
                                 (0,  0, 1)))

        c = np.cos(np.deg2rad(-yaw))
        s = np.sin(np.deg2rad(-yaw))
        yaw_matrix = np.array(((1, 0,  0),
                               (0, c, -s),
                               (0, s,  c)))
        yaw_matrix = np.array(((1, 0,  0),
                               (0, 1,  0),
                               (0, 0,  1)))
        rotation_matrix = np.dot(yaw_matrix, pitch_matrix)
        location = np.array((0, 0, distance))
        print(location)
        location = np.dot(yaw_matrix, location)
        print(yaw_matrix)
        print(location)
        location = np.dot(pitch_matrix, location)
        print(pitch_matrix)
        print(location)
        final_location = location + self.location
        print(final_location)

        return "c 0 240 300 " + \
        " %s %s %s 35.0 35.0 35.0 "%(0, -1, 0) + \
        ' '.join(str(e) for e in self.resolution) + " "
        # epsilon = 0.000001
        # return "c " + ' '.join(str(e) for e in final_location) + " " + \
        # " %s %s %s 35.0 35.0 35.0 "%(-location[0] + epsilon, -location[1] + epsilon, location[2] + epsilon) + \
        # ' '.join(str(e) for e in self.resolution) + " "

class Lit:
    def __init__(self, loc, intens=19000):
        if len(loc) != 3:
            raise ValueError('Light\'s constructor accepts: [(x,y,z),i]')
        self.intensity = intens
        self.location = loc

    def __str__(self):
        return "l p " + ' '.join(str(e) for e in self.location) + (" "+str(self.intensity))*3

class Mat:
    def __init__(self, col):
        if len(col) != 3:
            raise ValueError('Material\'s constructor accepts: (rgb)')
        self.color = col

    def __str__(self):
        return "m " + ' '.join(str(e) for e in self.color) + " 0.7 0.7 0.7 100. 0.5 0.5 0.5"


class Renderer:

    def __write_scene(self, view, light, object_prims,
                      background_prims, grid_prims, object_color, background_surface_color, grid_color, ambient_light_intensity):
        text_file = (str(Mat(object_color)) + NL)
        for p in object_prims:
            text_file += (str(p) + NL)
        text_file += (str(Mat(grid_color)) + NL)
        for p in grid_prims:
            text_file += (str(p) + NL)
        text_file += (str(Mat(background_surface_color)) + NL)
        for p in background_prims:
            text_file += (str(p) + NL)
        text_file += (str(light) + NL)
        text_file += ("l a" + (" " + str(ambient_light_intensity))*3 + NL)
        text_file += (view)
        return text_file


    def render(self, views, light, object_prims, background_prims, res_x, res_y, grid_prims=[], object_color=(1,0,0), \
               background_surface_color=(0.9, 0.9, 0.9), grid_color=(0.3,0.3,0.3), ambient_light_intensity=0.2):
        shadows = []
        noshadows = []
        for view in views:
            scene_text = self.__write_scene(view, light, object_prims, background_prims, grid_prims, \
                                    object_color, background_surface_color, grid_color, ambient_light_intensity)
            rend_output = renderer_lib.render(scene_text, res_x, res_y)
            shadow, noshadow = rend_output.reshape((2, res_x, res_y, 3))
            shadows.append(shadow)
            noshadows.append(noshadow)
        return np.concatenate(shadows, axis=1), np.concatenate(noshadows, axis=1)


