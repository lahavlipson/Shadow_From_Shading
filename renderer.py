import os
from scipy.ndimage import imread

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
    
    def __init__(self, exec_path, tmp_folder=None):
        self.number_of_scenes_written = 0
        if tmp_folder is not None:
            self.folder = tmp_folder
        else:
            if not os.path.isdir("tmp_scenes"):
                os.mkdir("tmp_scenes")
            self.folder = "tmp_scenes"
        self.executable = exec_path

    def __write_scene_file(self, scene_file_name, camera, light, object_prims, 
                           background_prims, object_color, background_surface_color, ambient_light_intensity):
        with open(scene_file_name, "w") as text_file: 
            text_file.write(str(Mat(object_color)) + NL)
            for p in object_prims:
                text_file.write(str(p) + NL)
            text_file.write(str(Mat(background_surface_color)) + NL)
            for p in background_prims:
                text_file.write(str(p) + NL)
            text_file.write(str(light) + NL)
            text_file.write("l a" + (" " + str(ambient_light_intensity))*3 + NL)
            text_file.write(str(camera))    
            

    def render(self, cameras, light, object_prims, background_prims, object_color=(1,0,0), \
               background_surface_color=(0.9, 0.9, 0.9), ambient_light_intensity=0.2):

        scene_folder_name = os.path.join(self.folder,"scene_" + str(self.number_of_scenes_written+1))
        if not os.path.isdir(scene_folder_name):
            os.mkdir(scene_folder_name)

        for i in range(len(cameras)):
            camera = cameras[i]
            print(len(cameras))
            scene_file_name = os.path.join(scene_folder_name, "scene" + SCN)
            output_file_names = os.path.join(scene_folder_name,"shadow_" + str(i) + PNG), \
                                os.path.join(scene_folder_name,"shadowless_" + str(i) + PNG)
            self.__write_scene_file(scene_file_name, camera, light, object_prims, background_prims, \
                                    object_color, background_surface_color, ambient_light_intensity)
            os.system(self.executable + " " + scene_file_name + " " + output_file_names[0] + " " + output_file_names[1])
        self.number_of_scenes_written += 1
        return imread(output_file_names[0]),imread(output_file_names[1])


