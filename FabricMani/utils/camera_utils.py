import numpy as np
import scipy.optimize as opt
import scipy
from scipy.spatial.distance import cdist


def build_depth_from_pointcloud(pointcloud, matrix_world_to_camera, imsize):
    """
    Build depth map from pointcloud

    Parameters:
    pointcloud
    matrix_world_to_camera
    imsize

    Returns:
    depth_2d: the depth map
    """
    # Analyze image dimensions
    height, width = imsize
    # Add homogeneous coordinates (fourth column all 1) to the point cloud, changing its shape from (n, 3) to (n, 4), for homogeneous matrix multiplication.
    pointcloud = np.concatenate([pointcloud, np.ones((len(pointcloud), 1))], axis=1)  # n x 4

    # World coordinate system to camera coordinate system:
    # matrix multiplication (4x4 @ 4xn → 4xn), then transpose to (n, 4), and take the first three columns to obtain the 3D camera coordinates (n, 3).
    camera_coordinate = matrix_world_to_camera @ pointcloud.T  # n x 4
    camera_coordinate = camera_coordinate.T  # n x 3
    # Compute the camera intrinsic matrix K based on the image size and field of view (fov) (45 degrees).
    K = intrinsic_from_fov(height, width, 45)

    # Extract principal point coordinates (image center) and focal length from intrinsic matrix K
    u0 = K[0, 2]    # Horizontal principal point (center along image width direction)
    v0 = K[1, 2]    # Vertical principal point (center along image height direction)
    fx = K[0, 0]    # Horizontal focal length
    fy = K[1, 1]    # Vertical focal length

    # Split the point cloud in camera coordinate system: x (horizontal), y (vertical), depth (depth, along the camera optical axis, i.e., the z-axis)
    # Caution：camera_coordinate is not the world coordinates of the camera, but the coordinates of the point cloud in the camera coordinate system.
    x, y, depth = camera_coordinate[:, 0], camera_coordinate[:, 1], camera_coordinate[:, 2]
    # Convert camera coordinates to pixel coordinates (pinhole camera model): u = (x*fx/depth) + u0, v = (y*fy/depth) + v0
    # np.rint() rounds to nearest integer
    # astype("int") converts to integer pixel coordinates
    u = np.rint((x * fx / depth + u0).astype("int"))
    v = np.rint((y * fy / depth + v0).astype("int"))

    # Flatten the pixel coordinate and depth arrays (ensure they are 1D for easier subsequent iteration)
    us = u.flatten()
    vs = v.flatten()
    depth = depth.flatten()

    # Initialize a dictionary
    # key = (u, v) pixel coordinates
    # values = depth corresponding to that pixel coordinate
    depth_map = dict()
    # Iterate over the pixel coordinates and depths corresponding the point cloud, and group the depth values by pixel coordinates.
    for u, v, d in zip(us, vs, depth):
        # If the pixel coordinates appear for the first time, initialize an empty list and add the current depth
        if depth_map.get((u, v)) is None:
            depth_map[(u, v)] = []
            depth_map[(u, v)].append(d)
        # Otherwise, directly append the depth value (a single pixel may correspond to multiple points in the point cloud)
        # Different points in 3D space project to the same pixel location on the 2D image plane.
        else:
            depth_map[(u, v)].append(d)

    # Initialize the depth map: with dimensions (height, width), with all initial values set to 0
    depth_2d = np.zeros((height, width))
    # Iterate over all pixel coordinates (by width and height)
    for u in range(width):
        for v in range(height):
            # If the pixel has a corresponding list of depth values
            if (u, v) in depth_map.keys():
                # Take the minimum of all depth values for that pixel as the final depth
                # (to solve the occlusion problem where multiple points project to the same pixel: keep the nearest one)
                depth_2d[v][u] = np.min(depth_map[(u, v)])

    return depth_2d


def pixel_coord_np(width, height):
    """
    Pixel in homogenous coordinate
    Returns:
        Pixel coordinate:       [3, width * height]
    """
    """
    Generate homogeneous coordinates for all pixels in the image, with output shape [3, width×height].
    
    Parameters:
    height (float): The height of the image (in pixels)
    width (float): The width of the image (in pixels)
    
    Returns:
    np.vstack: stack into a 2D matrix with 3 rows and N columns (N = width * height)
    """
    # 1. Generate pixel x-coordinates along the width direction: a 1D array ranging from 0 to width-1 (integers), length = width
    # Example (width=3): x = [0, 1, 2]
    x = np.linspace(0, width - 1, width).astype(np.int)
    # 2. Generate pixel y-coordinates along the height direction: a 1D array ranging from 0 to height-1 (integers), length = height
    # Example (height=2): y = [0, 1]
    y = np.linspace(0, height - 1, height).astype(np.int)
    # 3. Generate pixel mesh coordinates: convert the 1D x/y arrays into 2D mesh grids (covering all pixels)
    # Example (width=3, height=2):
    # x (2D) = [[0, 1, 2], [0, 1, 2]] (shape=(2,3))
    # y (2D) = [[0, 0, 0], [1, 1, 1]] (shape=(2,3))
    # Caution: The [x, y] are two 2D arrays, not a single 2D array.
    [x, y] = np.meshgrid(x, y)

    # 4. Assemble homogeneous coordinate matrix by stacking vertically:
    # - x.flatten(): flatten the 2D x array to 1D, example: [0,1,2,0,1,2] (shape=(6,))
    # - y.flatten(): flatten the 2D y array to 1D, example: [0,0,0,1,1,1] (shape=(6,))
    # - np.ones_like(...): generate an array of ones with the same length, example: [1,1,1,1,1,1] (shape=(6,))
    # - np.vstack: stack into a 2D matrix with 3 rows and N columns (N = width * height), example (width=3, height=2):
    # [[0 1 2 0 1 2]
    #  [0 0 0 1 1 1]
    #  [1 1 1 1 1 1]]
    return np.vstack((x.flatten(), y.flatten(), np.ones_like(x.flatten())))


def intrinsic_from_fov(height, width, fov=90):
    """
    Basic Pinhole Camera Model
    intrinsic params from fov and sensor width and height in pixels
    Returns:
        K:      [4, 4]
    """
    """
    Calculate the camera intrinsic matrix based on the given image height, width, and field of view.
    This function is based on the basic pinhole camera model. It calculates the horizontal focal length through the horizontal field of view,
    then derives the vertical field of view from the horizontal field of view and the aspect ratio of the image, and calculates the vertical focal length.
    Finally, it constructs the camera intrinsic matrix, which can be used to convert 3D points in the camera coordinate system to 2D points in the pixel coordinate system.

    Parameters:
    height (float): The height of the image (in pixels)
    width (float): The width of the image (in pixels)
    fov (float, optional): The horizontal field of view, with a default value of 90 degrees

    Returns:
    np.ndarray: An (4, 4) camera intrinsic matrix used to convert 3D points in the camera coordinate system to 2D points in the pixel coordinate system
    """

    # Calculate the principal point coordinates of the image (center of the pixel coordinate system):
    # px = half of the width, py = half of the height.
    # Example (width = 640, height = 480): px = 320.0, py = 240.0
    px, py = (width / 2, height / 2)

    # Convert the horizontal field of view (fov) from degrees to radians (rad):
    # hfov = degree value × π/180.
    # Example (fov = 90): hfov = 90/360×2×π = π/2 ≈ 1.5708 radians
    hfov = fov / 360. * 2. * np.pi

    # Calculate the horizontal focal length fx:
    # fx = image width / (2 × tan(half of the horizontal field of view)).
    # Example (width = 640, hfov = π/2): fx = 640/(2×tan(π/4)) = 320.0
    fx = width / (2. * np.tan(hfov / 2.))

    # Calculate the vertical field of view (vfov):
    # Derived from the horizontal field of view and the aspect ratio of the image.
    # vfov = 2 × arctan(tan(half of hfov) × height / width).
    # Example (width = 640, height = 480, hfov = π/2): vfov ≈ 1.176 radians
    vfov = 2. * np.arctan(np.tan(hfov / 2) * height / width)

    # Calculate the vertical focal length fy:
    # fy = image height / (2 × tan(half of the vertical field of view)).
    # Example (height = 480, vfov ≈ 1.176): fy ≈ 320.0
    fy = height / (2. * np.tan(vfov / 2.))

    # Construct the 4×4 camera intrinsic matrix K (homogeneous form):
    # The output K matrix (fx = 320, fy = 320, px = 320, py = 240):
    # K = [[320.  0.  320.  0.]
    #      [  0. 320. 240.  0.]
    #      [  0.   0.   1.  0.]
    #      [  0.   0.   0.  1.]]
    # This matrix is used to convert 3D points in the camera coordinate system to 2D points (u, v) in the pixel coordinate system.
    # Matrix formula: u = (K[0,0]×X)/Z + K[0,2], v = (K[1,1]×Y)/Z + K[1,2].
    # Example: Camera point (X = 100, Y = 50, Z = 200)
    # u = (320×100)/200 + 320 = 480, v = (320×50)/200 + 240 = 320
    return np.array([[fx, 0, px, 0.],
                     [0, fy, py, 0.],
                     [0, 0, 1., 0.],
                     [0., 0., 0., 1.]])


def get_rotation_matrix(angle, axis):
    """
    Generate a 4x4 rotation matrix based on the given rotation angle and axis.

    Parameters:
    - angle (float): The rotation angle in radians.
    - axis (numpy.ndarray): The axis of rotation. This vector will be normalized.

    Returns:
    - numpy.ndarray: A 4x4 rotation matrix in homogeneous coordinates, which can be used to rotate 3D vectors.
    """

    # Normalize the rotation axis to a unit vector
    axis = axis / np.linalg.norm(axis)

    # Calculate the sine and cosine of the rotation angle for use in matrix element calculations
    s = np.sin(angle)
    c = np.cos(angle)


    m = np.zeros((4, 4))

    m[0][0] = axis[0] * axis[0] + (1.0 - axis[0] * axis[0]) * c
    m[0][1] = axis[0] * axis[1] * (1.0 - c) - axis[2] * s
    m[0][2] = axis[0] * axis[2] * (1.0 - c) + axis[1] * s
    m[0][3] = 0.0

    m[1][0] = axis[0] * axis[1] * (1.0 - c) + axis[2] * s
    m[1][1] = axis[1] * axis[1] + (1.0 - axis[1] * axis[1]) * c
    m[1][2] = axis[1] * axis[2] * (1.0 - c) - axis[0] * s
    m[1][3] = 0.0

    m[2][0] = axis[0] * axis[2] * (1.0 - c) - axis[1] * s
    m[2][1] = axis[1] * axis[2] * (1.0 - c) + axis[0] * s
    m[2][2] = axis[2] * axis[2] + (1.0 - axis[2] * axis[2]) * c
    m[2][3] = 0.0

    m[3][0] = 0.0
    m[3][1] = 0.0
    m[3][2] = 0.0
    m[3][3] = 1.0

    return m


def get_world_coords(rgb, depth, env, particle_pos=None):
    height, width, _ = rgb.shape
    K = intrinsic_from_fov(height, width, 45)  # the fov is 90 degrees

    # Apply back-projection: K_inv @ pixels * depth
    u0 = K[0, 2]
    v0 = K[1, 2]
    fx = K[0, 0]
    fy = K[1, 1]

    x = np.linspace(0, width - 1, width).astype(float)
    y = np.linspace(0, height - 1, height).astype(float)
    u, v = np.meshgrid(x, y)
    one = np.ones((height, width, 1))
    x = (u - u0) * depth / fx
    y = (v - v0) * depth / fy
    z = depth
    cam_coords = np.dstack([x, y, z, one])

    matrix_world_to_camera = get_matrix_world_to_camera(
        env.camera_params[env.camera_name]['pos'], env.camera_params[env.camera_name]['angle'])

    # convert the camera coordinate back to the world coordinate using the rotation and translation matrix
    cam_coords = cam_coords.reshape((-1, 4)).transpose()  # 4 x (height x width)
    world_coords = np.linalg.inv(matrix_world_to_camera) @ cam_coords  # 4 x (height x width)
    world_coords = world_coords.transpose().reshape((height, width, 4))

    return world_coords


def get_observable_particle_index(world_coords, particle_pos, rgb, depth):
    height, width, _ = rgb.shape
    # perform the matching of pixel particle to real particle
    particle_pos = particle_pos[:, :3]

    estimated_world_coords = np.array(world_coords)[np.where(depth > 0)][:, :3]

    distance = scipy.spatial.distance.cdist(estimated_world_coords, particle_pos)
    # Each point in the point cloud will cover at most two particles. Particles not covered will be deemed occluded
    estimated_particle_idx = np.argpartition(distance, 2)[:, :2].flatten()
    estimated_particle_idx = np.unique(estimated_particle_idx)

    return np.array(estimated_particle_idx, dtype=np.int32)


def get_observable_particle_index_old(world_coords, particle_pos, rgb, depth):
    height, width, _ = rgb.shape
    # perform the matching of pixel particle to real particle
    particle_pos = particle_pos[:, :3]

    estimated_world_coords = np.array(world_coords)[np.where(depth > 0)][:, :3]

    distance = scipy.spatial.distance.cdist(estimated_world_coords, particle_pos)

    estimated_particle_idx = np.argmin(distance, axis=1)

    estimated_particle_idx = np.unique(estimated_particle_idx)

    return np.array(estimated_particle_idx, dtype=np.int32)


def get_observable_particle_index_3(pointcloud, mesh, threshold=0.0216):
    ### bi-partite graph matching
    distance = scipy.spatial.distance.cdist(pointcloud, mesh)
    if pointcloud.shape[0] > mesh.shape[0]:
        column_idx = np.argmin(distance, axis=1)
    else:
        distance[distance > threshold] = 1e10
        row_idx, column_idx = opt.linear_sum_assignment(distance)

    distance_mapped = distance[np.arange(len(pointcloud)), column_idx]
    pointcloud = pointcloud[distance_mapped < threshold]
    column_idx = column_idx[distance_mapped < threshold]

    return pointcloud, column_idx


def get_mapping_from_pointcloud_to_partile_nearest_neighbor(pointcloud, particle):
    distance = scipy.spatial.distance.cdist(pointcloud, particle)
    nearest_idx = np.argmin(distance, axis=1)
    return nearest_idx


def get_observable_particle_index_4(pointcloud, mesh, threshold=0.0216):
    # perform the matching of pixel particle to real particle
    estimated_world_coords = pointcloud

    distance = scipy.spatial.distance.cdist(estimated_world_coords, mesh)
    estimated_particle_idx = np.argmin(distance, axis=1)

    return pointcloud, np.array(estimated_particle_idx, dtype=np.int32)


def get_observable_particle_pos(world_coords, particle_pos):
    # perform the matching of pixel particle to real particle
    particle_pos = particle_pos[:, :3]
    distance = scipy.spatial.distance.cdist(world_coords, particle_pos)
    estimated_particle_idx = np.argmin(distance, axis=1)
    observable_particle_pos = particle_pos[estimated_particle_idx]

    return observable_particle_pos


def get_matrix_world_to_camera(cam_pos=[-0.0, 0.82, 0.82], cam_angle=[0, -45 / 180. * np.pi, 0.]):
    cam_x, cam_y, cam_z = cam_pos[0], cam_pos[1], \
                          cam_pos[2]
    cam_x_angle, cam_y_angle, cam_z_angle = cam_angle[0], cam_angle[1], \
                                            cam_angle[2]

    # get rotation matrix: from world to camera
    matrix1 = get_rotation_matrix(- cam_x_angle, [0, 1, 0])
    matrix2 = get_rotation_matrix(- cam_y_angle - np.pi, [1, 0, 0])
    rotation_matrix = matrix2 @ matrix1

    # get translation matrix: from world to camera
    translation_matrix = np.zeros((4, 4))
    translation_matrix[0][0] = 1
    translation_matrix[1][1] = 1
    translation_matrix[2][2] = 1
    translation_matrix[3][3] = 1
    translation_matrix[0][3] = - cam_x
    translation_matrix[1][3] = - cam_y
    translation_matrix[2][3] = - cam_z

    return rotation_matrix @ translation_matrix


def project_to_image(matrix_world_to_camera, world_coordinate, height=360, width=360):
    world_coordinate = np.concatenate([world_coordinate, np.ones((len(world_coordinate), 1))], axis=1)  # n x 4
    camera_coordinate = matrix_world_to_camera @ world_coordinate.T  # 3 x n
    camera_coordinate = camera_coordinate.T  # n x 3
    K = intrinsic_from_fov(height, width, 45)  # the fov is 90 degrees

    u0 = K[0, 2]
    v0 = K[1, 2]
    fx = K[0, 0]
    fy = K[1, 1]

    x, y, depth = camera_coordinate[:, 0], camera_coordinate[:, 1], camera_coordinate[:, 2]
    u = (x * fx / depth + u0).astype("int")
    v = (y * fy / depth + v0).astype("int")

    return u, v


def _get_depth(matrix, vec, height):
    """ Get the depth such that the back-projected point has a fixed height"""
    return (height - matrix[1, 3]) / (vec[0] * matrix[1, 0] + vec[1] * matrix[1, 1] + matrix[1, 2])


def get_world_coor_from_image(u, v, image_size, matrix_world_to_camera, all_depth):
    height, width = image_size
    K = intrinsic_from_fov(height, width, 45)  # the fov is 90 degrees

    matrix = np.linalg.inv(matrix_world_to_camera)

    u0, v0, fx, fy = K[0, 2], K[1, 2], K[0, 0], K[1, 1]

    depth = all_depth[v][u]
    if depth == 0:
        vec = ((u - u0) / fx, (v - v0) / fy)
        depth = _get_depth(matrix, vec, 0.00625)  # Height to be the particle radius

    x = (u - u0) * depth / fx
    y = (v - v0) * depth / fy
    z = depth
    cam_coords = np.array([x, y, z, 1])
    cam_coords = cam_coords.reshape((-1, 4)).transpose()  # 4 x (height x width)

    world_coord = matrix @ cam_coords  # 4 x (height x width)
    world_coord = world_coord.reshape(4)
    return world_coord[:3]


def get_target_pos(pos, u, v, image_size, matrix_world_to_camera, depth):
    coor = get_world_coor_from_image(u, v, image_size, matrix_world_to_camera, depth)
    dists = cdist(coor[None], pos)[0]
    idx = np.argmin(dists)
    return pos[idx] + np.array([0, 0.01, 0])
