import os
import renderer_lib

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
    def __init__(self, loc, direc, res):
        if len(loc) != 3 or len(direc) != 3 or len(res) != 2:
            raise ValueError('Camera\'s constructor accepts: (x,y,z),(x,y,z),(h,w)')
        self.location = loc
        self.direction = direc
        self.resolution = res
        
    def __str__(self):
        return "c " + ' '.join(str(e) for e in self.location) + " " + \
        ' '.join(str(e) for e in self.direction) + " 35.0 35.0 25.0 " + \
        ' '.join(str(e) for e in self.resolution) + " "
        
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

    def __write_scene(self, camera, light, object_prims,
                      background_prims, object_color, background_surface_color, ambient_light_intensity):
        text_file = (str(Mat(object_color)) + NL)
        for p in object_prims:
            text_file += (str(p) + NL)
        text_file += (str(Mat(background_surface_color)) + NL)
        for p in background_prims:
            text_file += (str(p) + NL)
        text_file += (str(light) + NL)
        text_file += ("l a" + (" " + str(ambient_light_intensity))*3 + NL)
        text_file += (str(camera))
        return text_file
            

    def render(self, cameras, light, object_prims, background_prims, object_color=(1,0,0), \
               background_surface_color=(0.9, 0.9, 0.9), ambient_light_intensity=0.2, name="output"):
        output = []
        for camera in cameras:
            scene_text = self.__write_scene(camera, light, object_prims, background_prims, \
                                    object_color, background_surface_color, ambient_light_intensity)
            rend_output = renderer_lib.render(scene_text, 480, 640)
            shadow, noshadow = rend_output.reshape((2, 480, 640, 3))
            output.append((shadow, noshadow))
        return output


