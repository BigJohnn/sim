import genesis as gs

# import pdb
gs.init(backend=gs.metal)

scene = gs.Scene(show_viewer=True)
plane = scene.add_entity(gs.morphs.Plane())
franka = scene.add_entity(
    gs.morphs.URDF(file='../sim/sim/resources/stompymicro/robot.urdf', pos=(0,0,0.3), fixed=True),
)

scene.build()


def run_sim(scene,):
    while True:
        scene.step()
        
gs.tools.run_in_another_thread(fn=run_sim, args=(scene,)) # start the simulation in another thread
scene.viewer.start() # start the viewer in the main thread (the render thread)