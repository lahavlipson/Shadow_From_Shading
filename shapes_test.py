from shapes import Sphere
from renderer import Renderer, Cam, Lit, Tri

rend = Renderer("./Renderer")

sp = Sphere((0, 0, 0), 25)
shapes = sp.render()
print(shapes)

background_prims = []
background_prims.append(Tri([(-1000.00,-40.00,1000.00), (1000.00,-40.00, 1000.00), (-1000.00,-40.00,-1000.00)]))
background_prims.append(Tri([(-1000.00,-40.00,-1000.00), (1000.00,-40.00, 1000.00), (1000.00,-40.00,-1000.00)]))


light = Lit((125,300,35),79000)
camera = Cam((200,222,83),( -.5,-.7,-.5), (640,480))
shadow_numpy_arr, shadowless_numpy_arr = rend.render(camera, light, shapes, background_prims)