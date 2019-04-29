from utils.scene import Scene
import cv2 as cv
import os


sc = Scene((20,80), True, gridlines_width=20, gridlines_spacing=30)
for _ in range(1):
    sc.add_object()
sc.ground_mesh()
sc.refocus_camera()
sc.mutate_all_objects()
sc.ground_mesh()
sc.refocus_camera()

shadows, noshadows = sc.render()
if not os.path.isdir("tmp_scenes"):
    os.mkdir("tmp_scenes")
cv.imwrite(os.path.join("tmp_scenes","shadows.png"), shadows)
cv.imwrite(os.path.join("tmp_scenes","noshadows.png"), noshadows)
