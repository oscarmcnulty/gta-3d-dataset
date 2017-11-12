from matplotlib import pyplot as plt
# from sklearn import linear_model
import numpy as np
# import sys
import math
from itertools import product, combinations
import json
import skimage.io
import cv2
# import itertools
import time
from scipy.stats import binned_statistic as bstat


def euler_to_dcm(theta):
    R_x = np.array([[1, 0, 0],
                    [0, math.cos(theta[0]), -math.sin(theta[0])],
                    [0, math.sin(theta[0]), math.cos(theta[0])]
                    ])

    R_y = np.array([[math.cos(theta[1]), 0, math.sin(theta[1])],
                    [0, 1, 0],
                    [-math.sin(theta[1]), 0, math.cos(theta[1])]
                    ])

    R_z = np.array([[math.cos(theta[2]), -math.sin(theta[2]), 0],
                    [math.sin(theta[2]), math.cos(theta[2]), 0],
                    [0, 0, 1]
                    ])

    R = np.dot(R_z, np.dot(R_y, R_x))

    return R

def rot_vec_to_dcm(rot_vec):
    theta = np.linalg.norm(rot_vec)
    rot_axis = rot_vec / theta

    K = np.array([[0,              -rot_axis[2],    rot_axis[1]],
                  [rot_axis[2],    0,               -rot_axis[0]],
                  [-rot_axis[1],   rot_axis[0],     0]])

    I = np.identity(3)

    R = I + math.sin(theta) * K + (1 - math.cos(theta)) * np.dot(K, K)

    return R

class GTAVehicle:

    """
    class_types_to_few = {
        u'SUVs': 'car',
        u'Sedans': 'car',
        u'Cycles': None,
        u'Motorcycles': 'bike',
        u'Service': 'truck',
        u'Coupes': 'car',
        u'Commercial': 'truck',
        u'Vans': 'truck',
        u'Compacts': 'car',
        u'Sports': 'car',
        u'SportsClassics': 'car',
        u'OffRoad': 'car',
        u'Trains': None,
        u'Muscle': 'car',
        u'Super': 'car',
        u'Emergency': 'truck',
        u'Utility': 'truck',
        u'Industrial': 'truck'
    }
    """

    class_types_to_few = {
        u'SUVs': 'car',
        u'Sedans': 'car',
        u'Cycles': None,
        u'Motorcycles': None,
        u'Service': 'car',
        u'Coupes': 'car',
        u'Commercial': 'car',
        u'Vans': 'car',
        u'Compacts': 'car',
        u'Sports': 'car',
        u'SportsClassics': 'car',
        u'OffRoad': 'car',
        u'Trains': None,
        u'Muscle': 'car',
        u'Super': 'car',
        u'Emergency': 'car',
        u'Utility': 'car',
        u'Industrial': 'car',
        u'Helicopters': None,
        u'Boats': None,
    }

    front_classes = ['car_front_0_to_45',
                     'car_front_45_to_90',
                     'car_front_90_to_135',
                     'car_front_135_to_180',
                     'car_front_180_to_225',
                     'car_front_225_to_270',
                     'car_front_270_to_315',
                     'car_front_315_to_0',
                     ]
    """
    birdseye_classes = ['car_birdseye_0_to_45',
                        'car_birdseye_45_to_90',
                        'car_birdseye_90_to_135',
                        'car_birdseye_135_to_180',
                        # 'car_birdseye_180_to_225',
                        # 'car_birdseye_225_to_270',
                        # 'car_birdseye_270_to_315',
                        # 'car_birdseye_315_to_0',
                        ]
    """
    birdseye_classes = ['car_birdseye_0_to_22',
                        'car_birdseye_22_to_45',
                        'car_birdseye_45_to_77',
                        'car_birdseye_77_to_90',
                        'car_birdseye_90_to_112',
                        'car_birdseye_112_to_135',
                        'car_birdseye_135_to_157',
                        'car_birdseye_157_to_180',
                        ]

    def __init__(self, vehicle, V, P):
        self.bbox_min = vehicle['bboxMin']
        self.bbox_max = vehicle['bboxMax']

        self.corners = np.array(list(product(
            [self.bbox_min[0], self.bbox_max[0]],
            [self.bbox_min[1], self.bbox_max[1]],
            [self.bbox_min[2], self.bbox_max[2]],
        )))

        self.is_visible = True

        self.V = V
        self.P = P

        # Initialise vairables for lazy evaluations
        self.bbox_oriented_birdseye = None
        self.bbox_birdseye = None

        self.rot = euler_to_dcm(np.array(vehicle['rot']) * math.pi / 180.0)
        self.pos = np.array([vehicle['pos']]).T

        self.class_type = GTAVehicle.class_types_to_few[vehicle["classType"]]

        pos_view = np.dot(V, np.concatenate((self.pos, [[1,]]), axis=0))

        # width = 1680
        # height = 1050
        # crop_width = 1680
        # crop_height = 550
        # height_offset = 200  # from top of image

        # object must be positioned such that -70 < z <0
        if not (-70 < pos_view[2, 0] < 0) or not (-35 < pos_view[0, 0] < 35):
            self.is_visible = False
        else:
            self.bbox_2d = self.get_bbox_2d()
            x_within = (0 < self.bbox_2d[0] < 1) or \
                       (0 < self.bbox_2d[2] < 1)
            y_within = (0 < self.bbox_2d[1] < 1) or \
                       (0 < self.bbox_2d[3] < 1)
            if x_within and y_within:
                self.is_visible = True
            else:
                self.is_visible = False

    def get_vehicle_base_height(self):
        birdseye_corners = np.array(list(product(
            [self.bbox_min[0], self.bbox_max[0]],
            [self.bbox_min[1], self.bbox_max[1]],
            [self.bbox_min[2], self.bbox_min[2]],
        )))

        corner_world = np.dot(self.rot, birdseye_corners.T) + self.pos
        corner_world_ = np.append(corner_world, [[1, 1, 1, 1, 1, 1, 1, 1]], axis=0)

        corner_view_ = np.dot(self.V, corner_world_).T

        corner_view = np.divide(corner_view_[:, :3],
                                corner_view_[:, 3:4])

        # Rotate to be axis aligned with view y-axis
        plane_normal_vec = np.cross(corner_view[4, :] - corner_view[0, :],
                                    corner_view[2, :] - corner_view[0, :])
        plane_normal_vec = plane_normal_vec / np.linalg.norm(plane_normal_vec)
        rot_vec = np.cross(plane_normal_vec, [0,1,0])

        rot_mat = rot_vec_to_dcm(rot_vec)

        centroid = np.mean(corner_view, axis=0)

        corner_view = np.dot(rot_mat, (corner_view - centroid).T).T + centroid

        return corner_view[(0,2,6,4), :], self.corners[1, 2] - self.corners[0, 2]

    def get_bbox_2d(self):
        # 1) project corners into world space
        # 2) project corners into view space
        # 3) project corners into clip space
        # 4) get min and max on x and y axes

        corner_world = np.dot(self.rot, self.corners.T) + self.pos
        corner_world_ = np.append(corner_world, [[1, 1, 1, 1, 1, 1, 1, 1,]], axis=0)

        corner_clip = np.dot(self.P, np.dot(self.V, corner_world_)).T

        corner_clip = np.divide(corner_clip[:, :3],
                                corner_clip[:, 3:4])

        # width = 1680
        # height = 1050
        # crop_width = 1680
        # crop_height = 550
        # height_offset = 300  # from top of image
        # coordintes are measure from top left of image

        xmin = (np.min(corner_clip.T[0]) + 1) / 2.
        xmax = (np.max(corner_clip.T[0]) + 1) / 2.
        ymin = (np.min(-corner_clip.T[1]) + 1) / 2. * 1050./550 - 300./550
        ymax = (np.max(-corner_clip.T[1]) + 1) / 2. * 1050./550 - 300./550

        return [xmin, ymin, xmax, ymax]

    def get_bbox_oriented_birdseye(self):
        """return a 4x2 array of the bbox corners of the vehicle on interval [0,1)
        """
        if self.bbox_oriented_birdseye is not None:
            return self.bbox_oriented_birdseye

        bbox, _ = self.get_vehicle_base_height()

        bbox = (bbox[:, (0, 2)] + [[35., 70.]]) / 70.

        self.bbox_oriented_birdseye = bbox

        return bbox

    def get_bbox_birdseye(self):
        if self.bbox_birdseye is not None:
            return self.bbox_birdseye

        if self.bbox_oriented_birdseye is None:
            self.get_bbox_oriented_birdseye()

        xmin = np.min(self.bbox_oriented_birdseye[:,0])
        xmax = np.max(self.bbox_oriented_birdseye[:,0])
        ymin = np.min(self.bbox_oriented_birdseye[:,1])
        ymax = np.max(self.bbox_oriented_birdseye[:,1])

        self.bbox_birdseye = [xmin, ymin, xmax, ymax]

        return self.bbox_birdseye

    def get_3d_bbox_edges(self):

        edges = []

        for s, e in combinations(self.corners, 2):  # s=start, e=end
            if np.sum(s == e) == 2:
                corner_world = np.dot(self.rot, np.array([s, e]).T) + self.pos
                corner_world_ = np.append(corner_world, [[1, 1]], axis=0)
                edges.append(
                    np.dot(self.V, corner_world_)[:3, :].T
                )

        return edges

    def get_front_class_idx(self):
        if self.class_type is None:
            return None

        if self.bbox_oriented_birdseye is None:
            self.get_bbox_oriented_birdseye()

        b = self.bbox_oriented_birdseye

        # 0->1 points forward with the vehicle
        # image coordinates are x to right, y down
        # calculate angels as x is up, y is left

        """
        Car birdseye bbox labelling:

        1___2
        | ^ |
        | | |
        |___|
        0   3
        """

        # vehicle direction in view space
        vehicle_angle = math.atan2( -(b[1, 0] - b[0, 0]), -(b[1, 1] - b[0, 1]) )

        # vehicle direction relative to view point
        view_angle = vehicle_angle - math.atan2( -(np.mean(b[:, 0]) - 0.5), 1. - np.mean(b[:, 1]) )

        idx = int((view_angle + 4.*math.pi) % (2*math.pi) / (math.pi/4.))

        """ use 8 classes instead of 4
        idx = int((view_angle + 4.*math.pi + math.pi/4.) % (2*math.pi) / (math.pi/2.))
        #  \ 1 /
        # 2 \ / 4
        #   / \
        #  / 3 \

        a = ['car_315_to_45',
             'car_45_to_135',
             'car_135_to_225',
             'car_225_to_315',]
        """
        return idx

    def get_front_class(self):
        if self.class_type is None:
            return None

        return GTAVehicle.front_classes[self.get_front_class_idx()]

    def get_birdseye_class(self):
        if self.class_type is None:
            return None

        if self.bbox_oriented_birdseye is None:
            self.get_bbox_oriented_birdseye()

        b = self.bbox_oriented_birdseye

        # 0->1 points forward with the vehicle
        # image coordinates are x to right, y down
        # convert to x is up, y is left

        # vehicle direction in view space
        vehicle_angle = math.atan2( -(b[1, 0] - b[0, 0]), -(b[1, 1] - b[0, 1]) )

        # don't care about forward/back heading of vehicle, just angle of bbox
        idx = int((vehicle_angle + 4.*math.pi) % (2*math.pi) / (math.pi/8.)) % 8
        # 45\ 0|  /  Measured CCW
        #  __\ | /__
        # 90 / | \
        #   /  |  \

        return GTAVehicle.birdseye_classes[idx]


class GTAData:
    height_min = -3.
    height_max = 1.5
    density_max = 2000000
    zrange_max = 1.

    def __init__(self, filename):
        d = json.loads(open(filename + '.json', 'r').read())

        self.filename = filename

        self.W = np.array(d['worldMatrix']['Values']).reshape(4, 4).T
        self.V = np.array(d['viewMatrix']['Values']).reshape(4, 4).T
        self.P = np.array(d['projectionMatrix']['Values']).reshape(4, 4).T

        # self.PV = np.dot(projection_m, view_m)

        self.c_pos = np.array(d['cameraPos'])
        self.c_rot = euler_to_dcm(np.array(d['cameraRot']))
        self.c_fov = d['cameraFOV']

        self.vehicles = []

        for v in d['vehicles']:
            new = GTAVehicle(v, self.V, self.P)
            if new.is_visible:
                self.vehicles.append(new)

        self.cloud_view = None
        self.height_map = None
        self.density_map = None
        self.zrange_map = None
        self.birdseye_maps = None
        self.depth_map = None

    def load_depth(self, bbox=None):
        """
        Load 3D point cloud data by transforming depth buffer to 3D view space
        """
        if self.cloud_view is not None and bbox is None:
            return self.cloud_view

        if self.depth_map is None:
            # depth 0, stencil 1
            self.depth_map = skimage.io.imread(self.filename + '.tiff', series=1)[0]

        x = np.linspace(-1, 1, self.depth_map.shape[1])
        y = np.linspace(1, -1, self.depth_map.shape[0])
        xv, yv = np.meshgrid(x, y)

        # in shape [layer={x,y,z}, row_idx, col_idx]
        cloud_clip = np.concatenate(([xv], [yv], [self.depth_map]), axis=0)

        # reshape to [row_idx, col_idx, layer={x,y,z}]
        if bbox is None:
            cloud_clip = np.rollaxis(cloud_clip, 0, 3)
        else:
            cloud_clip = np.rollaxis(
                cloud_clip[:, bbox[1]:(bbox[3]+1), bbox[0]:(bbox[2]+1)], 0, 3)

        # reshape as [sample_idx, layer={x,y,z}]
        cloud_clip = np.reshape(cloud_clip,
                                (cloud_clip.shape[0] * cloud_clip.shape[1], 3))

        # add ones for affine transformation
        cloud_clip = np.concatenate((cloud_clip,
                                     np.ones((cloud_clip.shape[0], 1))), axis=1)

        # transform point cloud from clip space to view/camera space
        cloud_view = np.dot(np.linalg.inv(self.P), cloud_clip.T).T

        # divide view space position by last element of view space projection
        cloud_view = np.divide(cloud_view[:, :3],
                                    cloud_view[:, 3:4])

        if bbox is None:
            self.cloud_view = cloud_view

        return cloud_view

    def load_depth_birdseye(self, bbox):
        if self.cloud_view is None:
            self.load_depth()

        return self.cloud_view[np.logical_and(
                np.all(self.cloud_view > np.array([[bbox[0], GTAData.height_min, bbox[1]]]), axis=1),
                np.all(self.cloud_view < np.array([[bbox[2], GTAData.height_max, bbox[3]]]), axis=1)
            ), 1]

    def get_birdseye_maps(self):
        """
        Lazily load the height map built from 3D point cloud extracted from
        depth buffer.
        """
        map_size = (512, 512)
        # tic = time.time()
        if self.birdseye_maps is not None:
            return self.birdseye_maps

        if self.cloud_view is None:
            self.load_depth()

        d = self.cloud_view[np.logical_and(
                np.all(self.cloud_view > np.array([[-35., GTAData.height_min, -70.]]), axis=1),
                np.all(self.cloud_view < np.array([[35., GTAData.height_max, 0.]]), axis=1)
            ), :]

        height = np.zeros(map_size)
        density = np.zeros(map_size)
        zrange = np.zeros(map_size)

        # d is of shape (points, 3)

        d = (np.dot([[0, 1000., 0],
                    [0, 0, 512./70],
                    [512./70, 0, 0]], d.T).T + \
            np.array([-GTAData.height_min*1000., 512., 256])).astype(np.uint16)

        order = np.lexsort(d.T)
        d = d[order, :]

        u, c = np.unique(d[:, (1, 2)], return_counts=True, axis=0)

        density[u[:, 1], u[:, 0]] = c

        index = np.empty(d.shape[0], 'bool')
        index[-1] = True
        index[:-1] = np.any(d[1:, (1, 2)] != d[:-1, (1, 2)], axis=1)

        min_index = np.empty(d.shape[0], 'bool')
        min_index[0] = True
        min_index[1:] = np.any(d[1:, (1, 2)] != d[:-1, (1, 2)], axis=1)

        height[d[index, 2], d[index, 1]] = d[index, 0]/5000.

        zrange[d[min_index, 2], d[min_index, 1]] = \
            (d[index, 0] - d[min_index, 0])/1000.

        self.height_map = (height.T * 256).astype(np.uint8)
        self.zrange_map = (np.minimum(
            zrange.T,
            GTAData.zrange_max
        ) * 256. / GTAData.zrange_max).astype(np.uint8)

        # TODO: Proper end points for range
        x = np.linspace(-349.5, 349.5, 512)
        y = np.linspace(699.5, 0.5, 512)
        xv, yv = np.meshgrid(x, y)

        self.density_map = (np.minimum(
            density.T * (np.power(xv, 2.) + np.power(yv, 2.)),
            GTAData.density_max
        ) * 256. / GTAData.density_max).astype(np.uint8)

        self.birdseye_maps = \
            np.stack((self.height_map, self.density_map, self.zrange_map))\
            .transpose((1, 2, 0))

        return self.birdseye_maps

    def write_birdseye_maps(self):
        if self.birdseye_maps is None:
            self.get_birdseye_maps()

        skimage.io.imsave(
            self.filename + '_birdseye.jpg',
            self.birdseye_maps,
            quality=90)

    def load_rgb(self):
        color = skimage.io.imread(self.filename + '.tiff', series=0)

        return color
