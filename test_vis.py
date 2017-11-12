
from vispy import app, scene
import numpy as np

from gta import GTAData

filename = 'data/gta_test/3fd50f7b-658b-4ef4-bb17-dfc1f287def8_00000819'

data = GTAData(filename)

bbox_edges = []
for v in data.vehicles:
    bbox_edges.extend(v.get_3d_bbox_edges())

# Create a canvas with a 3D viewport
canvas = scene.SceneCanvas(keys='interactive', title='GTA Scene', bgcolor='white')
canvas.show()
view = canvas.central_widget.add_view()

# Add vehicle bounding box to view
if len(bbox_edges) > 0:
    scene.visuals.Line(pos=np.array(bbox_edges),
                       color=(0.2, 0.5, 0.3, 1),
                       connect='segments', parent=view.scene)

scene.visuals.XYZAxis(parent=view.scene)

# Add point cloud
c_pos_marker = scene.visuals.Markers(parent=view.scene)
cloud = np.array(data.load_depth())
cloud = cloud[np.all([-100,-100,-70] < cloud, axis=1), :]

c_pos_marker.set_data(pos=cloud, symbol='o', size=0.01,
                      edge_width=0, edge_width_rel=None,
                      edge_color='black', face_color='black',
                      scaling=False)

view.camera = scene.FlyCamera(fov=data.c_fov)
view.camera.auto_roll = False

view.camera.set_range((-150, 150), (-150, 150), (-150, 150))

app.run()
