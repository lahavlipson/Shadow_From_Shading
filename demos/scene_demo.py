from utils.scene import Scene
import cv2 as cv
import os


g = Scene(True, 3, 100)
g.add_object()
shadows, noshadows = g.render()
if not os.path.isdir("tmp_scenes"):
    os.mkdir("tmp_scenes")
cv.imwrite(os.path.join("tmp_scenes","shadows.png"), shadows)
cv.imwrite(os.path.join("tmp_scenes","noshadows.png"), noshadows)
