from shapes import Sphere, Cuboid, Tetrahedron
from renderer import Renderer, Cam, Lit, Tri

rend = Renderer("./Renderer")

sp = Sphere((0, 75, 0), 25)
shapes = sp.render()
cuboid = Cuboid((30, 25, -100))
cuboid.scale(25)
cuboid.rotate(45, 90)
shapes.extend(cuboid.render())
tetrahedron = Tetrahedron((100, 50, -100))
tetrahedron.scale(25)
tetrahedron.rotate(45, 90)
shapes.extend(tetrahedron.render())

background_prims = []
background_prims.append(Tri([(-1000.00,-40.00,1000.00), (1000.00,-40.00, 1000.00), (-1000.00,-40.00,-1000.00)]))
background_prims.append(Tri([(-1000.00,-40.00,-1000.00), (1000.00,-40.00, 1000.00), (1000.00,-40.00,-1000.00)]))


light = Lit((125,300,35),79000)
camera = Cam((200,222,83),( -.5,-.7,-.5), (640,480))
shadow_numpy_arr, shadowless_numpy_arr = rend.render(camera, light, shapes, background_prims)
