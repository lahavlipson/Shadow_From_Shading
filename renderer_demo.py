from renderer import Renderer, Tri, Cir, Cam, Lit


rend = Renderer("./Renderer")
object_prims = []
object_prims.append(Tri([(-40,0,20.0),(0,0,-20.0),(40,0,20)]))
object_prims.append(Tri([(0,40,0),(0,0,-20),(-40,0,20.0)]))
object_prims.append(Tri([(0,40,0.0),(-40,0,20.0),(40,0,20)]))
object_prims.append(Cir([(0,20,-80),30]))
object_prims.append(Cir([(55,20,-80),25]))
object_prims.append(Cir([(-55,20,-80),25]))
object_prims.append(Cir([(100,20,-80),20]))
object_prims.append(Cir([(-100,20,-80),20]))
object_prims.append(Cir([(135,20,-80),15]))
object_prims.append(Cir([(-135,20,-80),15]))
object_prims.append(Cir([(160,20,-80),10]))
object_prims.append(Cir([(-160,20,-80),10]))
object_prims.append(Cir([(175,20,-80),5]))
object_prims.append(Cir([(-175,20,-80),5]))

background_prims = []
background_prims.append(Tri([(-1000.00,-40.00,1000.00), (1000.00,-40.00, 1000.00), (-1000.00,-40.00,-1000.00)]))
background_prims.append(Tri([(-1000.00,-40.00,-1000.00), (1000.00,-40.00, 1000.00), (1000.00,-40.00,-1000.00)]))


light = Lit((125,300,35),79000)
camera = Cam((200,222,83),( -.5,-.7,-.5), (640,480))
shadow_numpy_arr, shadowless_numpy_arr = rend.render(camera, light,object_prims, background_prims)
