
import os
import utm
import math
import torch
import random
import imageio
import logging
import numpy as np
from PIL import Image
from PIL import ImageFile
import torchvision.transforms as T
from collections import defaultdict

from datasets.map_utils import create_map
import datasets.dataset_utils as dataset_utils

ImageFile.LOAD_TRUNCATED_IMAGES = True

PANO_WIDTH = int(512*6.5)


def get_angle(conv_point, obs_point):
    obs_e, obs_n = float(obs_point[0]), float(obs_point[1])
    conv_e, conv_n = conv_point
    side1 = conv_e - obs_e
    side2 = conv_n - obs_n
    angle = - math.atan2(side1, side2) / math.pi * 90 * 2
    return angle


def get_eigen_things(utm_coords):
    mu = utm_coords.mean(0)
    norm_data = utm_coords - mu
    eigenvectors, eigenvalues, v = np.linalg.svd(norm_data.T, full_matrices=False)
    projected_data = np.dot(utm_coords, eigenvectors)
    sigma = projected_data.std(0).mean()
    return eigenvectors, eigenvalues, mu, sigma


def rotate_2d_vector(vector, angle):
    assert vector.shape == (2,)
    theta = np.deg2rad(angle)
    rot_mat = np.array([[np.cos(theta), -np.sin(theta)],
                       [np.sin(theta), np.cos(theta)]])
    rotated_point = np.dot(rot_mat, vector)
    return rotated_point


def get_convergence_point(utm_coords, meters_from_center=20, angle=0):
    """Return the convergence point from a set of utm coords.
    Also return the eigen_ratio, which is the ratio between the largest and
    smallest eigenvalue. An eigen_ratio > 2 means that the coords are well
    aligned (likely in a straight road).
    """
    B, D = utm_coords.shape
    assert D == 2
    eigenvectors, eigenvalues, mu, sigma = get_eigen_things(utm_coords)

    direction = rotate_2d_vector(eigenvectors[1], angle)
    convergence_point = mu + direction * meters_from_center

    eigen_ratio = max(eigenvalues) / (min(eigenvalues) + 1e-6)
    return convergence_point, eigen_ratio


class EigenPlacesDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_folder,
                 M=20, N=5, conv_dist=10, current_group=0, 
                 min_images_per_class=10, angle=0, visualize_classes=0):
        """
        Parameters (please check our paper for a clearer explanation of the parameters).
        ----------
        dataset_folder : str, the path of the folder with the train images.
        M : int, the length of the side of each cell in meters.
        N : int, distance (M-wise) between two classes of the same group.
        conv_dist : int, distance (M-wise) between the center of the class and
            the convergence point. The center of the class is computed as the
            mean of the positions of the images within the class.
        min_eigen_ratio : int, within a class if the ratio between the bigger
            eigenvalue and the smaller eigenvalue (here called eigen_ratio) is
            smaller than min_eigen_ratio, the class is discarded. Low values of
            eigen_ratio happen in sparse classes (e.g. in a square), high
            values in linear classes (e.g. images from a straight road).
        current_group : int, which one of the groups to consider.
        min_images_per_class : int, minimum number of image in a class.
        """
        super().__init__()
        self.M = M
        self.N = N
        self.conv_dist = conv_dist
        self.current_group = current_group
        self.dataset_folder = dataset_folder
        
        filename = f"cache/sfxl_M{M}_N{N}_mipc{min_images_per_class}.torch"
        if not os.path.exists(filename):
            os.makedirs("cache", exist_ok=True)
            logging.info(f"Cached dataset {filename} does not exist, I'll create it now.")
            self.initialize(dataset_folder, M, N, min_images_per_class, filename)
        elif current_group == 0:
            logging.info(f"Using cached dataset {filename}")
        
        classes_per_group, self.images_per_class = torch.load(filename)
        if current_group >= len(classes_per_group):
            raise ValueError(f"With this configuration there are only {len(classes_per_group)} " +
                             f"groups, therefore I can't create the {current_group}th group. " +
                             "You should reduce the number of groups by setting for example " +
                             f"'--groups_num {current_group}'")
        self.classes_ids = classes_per_group[current_group]
        
        new_classes_ids = []
        self.conv_point_per_class = {}
        for class_id in self.classes_ids:
            paths = self.images_per_class[class_id]
            u_coords = np.array([p.split("@")[1:3] for p in paths]).astype(float)
                        
            convergence_point, eigen_ratio = get_convergence_point(u_coords, conv_dist,
                                                                   angle=angle)
            new_classes_ids.append(class_id)
            self.conv_point_per_class[class_id] = convergence_point

        self.classes_ids = new_classes_ids

        # This is only for logging, debugging and visualizations
        for class_num in range(visualize_classes):
            random_class_id = random.choice(self.classes_ids)
            paths = self.images_per_class[random_class_id]
            convergence_point = self.conv_point_per_class[random_class_id]
            conv_point_lat_lon = np.array(utm.to_latlon(convergence_point[0], convergence_point[1], 10, 'S'))
            lats_lons = np.array([p.split("@")[5:7] for p in paths]).astype(float)
            lats_lons += (np.random.randn(*lats_lons.shape) / 500000)  # Add a little noise to avoid overlapping

            min_e, min_n = random_class_id
            cell_utms = (min_e, min_n), (min_e, min_n + M), (min_e + M, min_n + M), (min_e + M, min_n)
            cell_corners = np.array([utm.to_latlon(*u, 10, 'S') for u in cell_utms])

            img = create_map([lats_lons, lats_lons.mean(0).reshape(1, 2), conv_point_lat_lon.reshape(1, 2), cell_corners],
                              colors=["r", "b", "g", "orange"],
                              legend_names=["images", "mean", "conv point", f"cell ({M} m)"],
                              dot_sizes=[10, 100, 100, 100])
            output_folder = os.path.dirname(logging.getLoggerClass().root.handlers[0].baseFilename)
            folder = f"{output_folder}/visualizations/group{current_group}_{class_num}_{random_class_id}"
            os.makedirs(folder)
            imageio.imsave(f"{folder}/@00_map.jpg", img)
            images_paths = self.images_per_class[random_class_id]
            for path in images_paths:
                crop = self.get_crop(self.dataset_folder + "/" + path, convergence_point)
                crop = T.functional.to_pil_image(crop)
                crop.save(f"{folder}/{os.path.basename(path)}")

    @staticmethod
    def get_crop(pano_path, convergence_point):
        obs_point = pano_path.split("@")[1:3]
        angle = - get_angle(convergence_point, obs_point) % 360
        crop_offset = int((angle / 360 * PANO_WIDTH) % PANO_WIDTH)
        yaw = int(pano_path.split("@")[9])
        north_yaw_in_degrees = (180-yaw) % 360
        yaw_offset = int((north_yaw_in_degrees / 360) * PANO_WIDTH)
        offset = (yaw_offset + crop_offset - 256) % PANO_WIDTH
        pano_tensor = T.functional.to_tensor(Image.open(pano_path))
        if pano_tensor.shape[2] <= offset + 512:
            pano_tensor = torch.cat([pano_tensor, pano_tensor], 2)
        crop = pano_tensor[:, :, offset : offset+512]
        return crop

    def __getitem__(self, class_num):
        # This function takes as input the class_num instead of the index of
        # the image. This way each class is equally represented during training.
        class_id = self.classes_ids[class_num]
        convergence_point = self.conv_point_per_class[class_id]
        pano_path = self.dataset_folder + "/" + random.choice(self.images_per_class[class_id])
        crop = self.get_crop(pano_path, convergence_point)
        return crop, class_num, pano_path
    
    def get_images_num(self):
        """Return the number of images within this group."""
        return sum([len(self.images_per_class[c]) for c in self.classes_ids])
    
    def __len__(self):
        """Return the number of classes within this group."""
        return len(self.classes_ids)
    
    @staticmethod
    def initialize(dataset_folder, M, N, min_images_per_class, filename):
        logging.debug(f"Searching training images in {dataset_folder}")
        
        images_paths = dataset_utils.read_images_paths(dataset_folder)
        logging.debug(f"Found {len(images_paths)} images")
        
        logging.debug("For each image, get its UTM east, UTM north from its path")
        images_metadatas = [p.split("@") for p in images_paths]
        # field 1 is UTM east, field 2 is UTM north
        utmeast_utmnorth = [(m[1], m[2]) for m in images_metadatas]
        utmeast_utmnorth = np.array(utmeast_utmnorth).astype(float)
        
        logging.debug("For each image, get class and group to which it belongs")
        class_id__group_id = [EigenPlacesDataset.get__class_id__group_id(*m, M, N)
                              for m in utmeast_utmnorth]
        
        logging.debug("Group together images belonging to the same class")
        images_per_class = defaultdict(list)
        for image_path, (class_id, _) in zip(images_paths, class_id__group_id):
            images_per_class[class_id].append(image_path)
        
        # Images_per_class is a dict where the key is class_id, and the value
        # is a list with the paths of images within that class.
        images_per_class = {k: v for k, v in images_per_class.items() if len(v) >= min_images_per_class}
        
        logging.debug("Group together classes belonging to the same group")
        # Classes_per_group is a dict where the key is group_id, and the value
        # is a list with the class_ids belonging to that group.
        classes_per_group = defaultdict(set)
        for class_id, group_id in class_id__group_id:
            if class_id not in images_per_class:
                continue  # Skip classes with too few images
            classes_per_group[group_id].add(class_id)
        
        # Convert classes_per_group to a list of lists.
        # Each sublist represents the classes within a group.
        classes_per_group = [list(c) for c in classes_per_group.values()]
        
        torch.save((classes_per_group, images_per_class), filename)
    
    @staticmethod
    def get__class_id__group_id(utm_east, utm_north, M, N):
        """Return class_id and group_id for a given point.
            The class_id is a triplet (tuple) of UTM_east, UTM_north 
            (e.g. (396520, 4983800)).
            The group_id represents the group to which the class belongs
            (e.g. (0, 1)), and it is between (0, 0) and (N, N).
        """
        rounded_utm_east = int(utm_east // M * M)  # Rounded to nearest lower multiple of M
        rounded_utm_north = int(utm_north // M * M)
        
        class_id = (rounded_utm_east, rounded_utm_north)
        # group_id goes from (0, 0) to (N, N)
        group_id = (rounded_utm_east % (M * N) // M,
                    rounded_utm_north % (M * N) // M)
        return class_id, group_id
