from shapes import Sphere, Cuboid, Tetrahedron
from renderer import Renderer, Cam, Lit, Tri
import cv2

rend = Renderer("")

sp = Sphere((0, 0, 0), 25)
shapes = sp.render()
cuboid = Cuboid((0, 0, -100))
cuboid.scale(25, axis=0)
cuboid.scale(50, axis=1)
cuboid.scale(75, axis=2)
cuboid.rotate(90, 90)
shapes.extend(cuboid.render())
tetrahedron = Tetrahedron((100, 0, 0))
tetrahedron.scale(25, axis=0)
tetrahedron.scale(50, axis=1)
tetrahedron.scale(20, axis=2)
tetrahedron.rotate(45, 180)
shapes.extend(tetrahedron.render())

background_prims = []
background_prims.append(Tri([(-1000.00,-40.00,1000.00), (1000.00,-40.00, 1000.00), (-1000.00,-40.00,-1000.00)]))
background_prims.append(Tri([(-1000.00,-40.00,-1000.00), (1000.00,-40.00, 1000.00), (1000.00,-40.00,-1000.00)]))


light = Lit((125,300,35),79000)
camera = [Cam((200,222,83),( -.5,-.7,-.5), (640,480))]
shadow, noshadow = rend.render(camera, light, shapes, background_prims)[0]
cv2.imwrite("shadow.png", shadow)
cv2.imwrite("noshadow.png", noshadow)
