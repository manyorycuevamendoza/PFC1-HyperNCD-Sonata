# _*_ coding: utf-8 _*_
"""
Time:     2023/8/1 16:58
Author:   Ruijie Xu
File:     dataset.py
"""
from copy import deepcopy
import os
import random

import MinkowskiEngine as ME
import numpy as np
import torch
import yaml
# from sklearn import cluster
import pickle
from utils.voxelizer import Voxelizer

# Carpeta donde se guardan los caches de regiones (.pkl). Por defecto, dentro del
# repo; se puede cambiar con la variable de entorno HYPERNCD_CACHE.
CACHE_ROOT = os.environ.get(
    "HYPERNCD_CACHE",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "region_save"),
)


def get_dataset(name):
    if name == "SemanticKITTI":
        return SemanticKITTIRestrictedDataset
    elif name == "SemanticPOSS":
        return SemanticPOSSRestrictedDataset
    elif name == "ArCH":
        return ArCHRestrictedDataset
    else:
        raise NameError(f'Dataset "{name}" not yet implemented')


class SemanticKITTIRestrictedDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            config_file="config/dataset.yaml",
            split="train",
            voxel_size=0.05,
            downsampling=80000,
            augment=False,
            label_mapping=None,
            data_splits=None,
            superpoint=False
    ):
        self.superpoint = superpoint
        self.data_splits = data_splits
        """Load data from given dataset directory."""

        with open(config_file, "r") as f:
            self.config = yaml.safe_load(f)

        self.files = {"input": []}
        if split != "test":
            self.files["label"] = []
        self.filenames = []

        self.voxel_size = voxel_size
        self.downsampling = downsampling
        self.augment = False
        if split == "train" and augment:
            self.augment = True
            self.clip_bounds = None
            self.scale_augmentation_bound = (0.95, 1.05)
            self.rotation_augmentation_bound = (
                (-np.pi / 20, np.pi / 20),
                (-np.pi / 20, np.pi / 20),
                (-np.pi / 20, np.pi / 20),
            )
            self.translation_augmentation_ratio_bound = None

            self.voxelizer = Voxelizer(
                voxel_size=self.voxel_size,
                clip_bound=self.clip_bounds,
                use_augmentation=self.augment,
                scale_augmentation_bound=self.scale_augmentation_bound,
                rotation_augmentation_bound=self.rotation_augmentation_bound,
                translation_augmentation_ratio_bound=self.translation_augmentation_ratio_bound,
                ignore_label=-1,
            )

        for sequence in self.config["split_sequence"][split]:
            for idx, type in enumerate(self.files.keys()):
                files_path = os.path.join(
                    self.config["dataset_path"],
                    "sequences",
                    sequence,
                    self.config["folder_name"][type],
                )
                #if not os.path.exists(files_path):
                #    raise RuntimeError("Point cloud directory missing: " + files_path)
                files = os.listdir(files_path)
                data = sorted([os.path.join(files_path, f) for f in files])
                if len(data) == 0:
                    raise RuntimeError("Missing data for " + type)
                self.files[type].extend(data)
                if idx == 0:
                    self.filenames.extend(data)

        self.num_files = len(self.filenames)

        if label_mapping is not None:
            self.label_mapping_function = np.vectorize(lambda x: label_mapping[x])
        else:
            self.label_mapping_function = None
        if data_splits == 3:
            self.unseen_id = 15
        else:
            self.unseen_id = 14

    def __len__(self):
        return self.num_files

    def __getitem__(self, t):

        pc_filename = self.files["input"][t]
        scan = np.fromfile(pc_filename, dtype=np.float32)
        scan = scan.reshape((-1, 4))
        coordinates = scan[:, 0:3]  # get xyz
        remissions = scan[:, 3]  # get remission
        features = np.ones((coordinates.shape[0], 1))


        import os
        if os.path.exists(CACHE_ROOT + "/KITTI/" + str(self.data_splits)+"/" + str(t) + '.pkl'):
            pass
        else:


            if "label" in self.files.keys():
                label_filename = self.files["label"][t]
                labels = np.fromfile(label_filename, dtype=np.int32)
                labels = labels.reshape((-1))
                labels = labels & 0xFFFF
                for index, element in enumerate(labels):
                    labels[index] = self.config["learning_map"].get(element, -1)
            else:
                    labels = np.negative(np.ones(coordinates.shape[0]))

            # REMOVE UNLABELED POINTS IF NOT IN TESTING
            if "label" in self.files.keys():
                labelled_idx = labels != -1
                labels1 = labels[labelled_idx]

            if self.label_mapping_function is not None:
                mapped_labels = self.label_mapping_function(labels1)
            else:
                mapped_labels = np.copy(labels)

            from sklearn import cluster
            if labels1[mapped_labels == self.unseen_id].shape[0] >= 1:
                clustering = cluster.DBSCAN(eps=0.2, min_samples=2).fit(coordinates[labels != -1][mapped_labels == self.unseen_id])
                region_id = clustering.labels_
            else:
                region_id = None

            save_path = CACHE_ROOT + "/KITTI/" + str(self.data_splits) + "/" + str(t) + '.pkl'
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

            with open(save_path, "wb") as f:
                pickle.dump(region_id, f)


        # AUGMENTATION
        if self.augment:
            # DOWNSAMPLING
            selected_idx = np.random.choice(
                coordinates.shape[0], self.downsampling, replace=False
            )
            selected_idx = np.sort(selected_idx)
            coordinates = coordinates[selected_idx]
            features = features[selected_idx]

            # TRANSFORMATIONS
            voxel_mtx, affine_mtx = self.voxelizer.get_transformation_matrix()

            rigid_transformation = affine_mtx @ voxel_mtx
            # Apply transformations

            homo_coords = np.hstack(
                (
                    coordinates,
                    np.ones((coordinates.shape[0], 1), dtype=coordinates.dtype),
                )
            )
            # coords = np.floor(homo_coords @ rigid_transformation.T[:, :3])
            coordinates = homo_coords @ rigid_transformation.T[:, :3]
        else:
            selected_idx = np.arange(coordinates.shape[0])

        if "label" in self.files.keys():
            label_filename = self.files["label"][t]
            labels = np.fromfile(label_filename, dtype=np.int32)
            labels = labels.reshape((-1))
            labels = labels & 0xFFFF



            for index, element in enumerate(labels):
                labels[index] = self.config["learning_map"].get(element, -1)
            sup_label = np.zeros_like(labels) - 1

            if self.superpoint:
                fr = open(CACHE_ROOT + '/KITTI/' + str(self.data_splits)+"/" + str(t) + '.pkl', "rb")
                superpoint = pickle.load(fr)
                fr.close()
                if superpoint is not None:
                    temp = sup_label[labels != -1]
                    temp[self.label_mapping_function(labels[labels != -1]) == self.unseen_id] = superpoint
                    sup_label[labels != -1] = temp

            if self.augment:
                labels = labels[selected_idx]
                sup_label = sup_label[selected_idx]

        else:
            labels = np.negative(np.ones(coordinates.shape[0]))

        # REMOVE UNLABELED POINTS IF NOT IN TESTING
        if "label" in self.files.keys():
            labelled_idx = labels != -1
            coordinates = coordinates[labelled_idx]
            features = features[labelled_idx]
            labels = labels[labelled_idx]
            selected_idx = selected_idx[labelled_idx]
            sup_label = sup_label[labelled_idx]
            if self.label_mapping_function is not None:
                mapped_labels = self.label_mapping_function(labels)
            else:
                mapped_labels = np.copy(labels)

        discrete_coords, unique_map = ME.utils.sparse_quantize(
            coordinates=coordinates,
            return_index=True,
            quantization_size=self.voxel_size,
        )

        unique_feats = features[unique_map]
        unique_labels = labels[unique_map]
        unique_mapped_labels = mapped_labels[unique_map]
        selected_idx = selected_idx[unique_map]
        unique_sup_label = sup_label[unique_map]
        selected_region_idx = np.unique(unique_sup_label)

        return (
            discrete_coords,
            unique_feats,
            unique_labels,
            selected_idx,
            unique_mapped_labels,
            unique_sup_label,
            selected_region_idx,
            t,
        )



class SemanticKITTIRestrictedDatasetCleanSplit(SemanticKITTIRestrictedDataset):
    def __init__(
            self,
            clean_mask,
            config_file="config/dataset.yaml",
            split="train",
            voxel_size=0.05,
            downsampling=80000,
            augment=False,
            label_mapping=None,
    ):
        super().__init__(
            config_file, split, voxel_size, downsampling, augment, label_mapping
        )
        self.filenames = np.array(self.filenames)[clean_mask]
        self.num_files = len(self.filenames)
        for key in self.files.keys():
            self.files[key] = np.array(self.files[key])[clean_mask]


class SemanticPOSSDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            config_file="config/semposs_dataset.yaml",
            split="train",
            voxel_size=0.05,
            downsampling=80000,
            augment=False,
    ):
        """Load data from given dataset directory."""

        with open(config_file, "r") as f:
            self.config = yaml.safe_load(f)

        self.files = {"input": []}
        if split != "test":
            self.files["label"] = []
        self.filenames = []

        self.voxel_size = voxel_size
        self.downsampling = downsampling
        self.augment = False
        if split == "train" and augment:
            self.augment = True
            self.clip_bounds = None
            self.scale_augmentation_bound = (0.95, 1.05)
            self.rotation_augmentation_bound = (
                (-np.pi / 20, np.pi / 20),
                (-np.pi / 20, np.pi / 20),
                (-np.pi / 20, np.pi / 20),
            )
            self.translation_augmentation_ratio_bound = None

            self.voxelizer = Voxelizer(
                voxel_size=self.voxel_size,
                clip_bound=self.clip_bounds,
                use_augmentation=self.augment,
                scale_augmentation_bound=self.scale_augmentation_bound,
                rotation_augmentation_bound=self.rotation_augmentation_bound,
                translation_augmentation_ratio_bound=self.translation_augmentation_ratio_bound,
                ignore_label=-1,
            )

        for sequence in self.config["split_sequence"][split]:
            for idx, type in enumerate(self.files.keys()):
                files_path = os.path.join(
                    self.config["dataset_path"],
                    "sequences",
                    sequence,
                    self.config["folder_name"][type],
                )
                if not os.path.exists(files_path):
                    raise RuntimeError("Point cloud directory missing: " + files_path)
                files = os.listdir(files_path)
                data = sorted([os.path.join(files_path, f) for f in files])
                if len(data) == 0:
                    raise RuntimeError("Missing data for " + type)
                self.files[type].extend(data)
                if idx == 0:
                    self.filenames.extend(data)

        learning_map = self.config["learning_map"]
        self.learning_map_function = np.vectorize(lambda x: learning_map[x])

        self.num_files = len(self.filenames)

    def __len__(self):
        return self.num_files

    def __getitem__(self, t):
        pc_filename = self.files["input"][t]
        scan = np.fromfile(pc_filename, dtype=np.float32)
        scan = scan.reshape((-1, 4))
        coordinates = scan[:, 0:3]  # get xyz
        remissions = scan[:, 3]  # get remission

        features = np.ones((coordinates.shape[0], 1))

        # AUGMENTATION
        if self.augment:
            # DOWNSAMPLING
            selected_idx = np.random.choice(
                coordinates.shape[0], self.downsampling, replace=False
            )
            coordinates = coordinates[selected_idx]
            features = features[selected_idx]

            # TRANSFORMATIONS
            voxel_mtx, affine_mtx = self.voxelizer.get_transformation_matrix()

            rigid_transformation = affine_mtx @ voxel_mtx
            # Apply transformations

            homo_coords = np.hstack(
                (
                    coordinates,
                    np.ones((coordinates.shape[0], 1), dtype=coordinates.dtype),
                )
            )
            # coords = np.floor(homo_coords @ rigid_transformation.T[:, :3])
            coordinates = homo_coords @ rigid_transformation.T[:, :3]
        else:
            selected_idx = np.arange(coordinates.shape[0])

        if "label" in self.files.keys():
            label_filename = self.files["label"][t]
            labels = np.fromfile(label_filename, dtype=np.int32)
            labels = labels.reshape((-1))
            labels = labels & 0xFFFF
            if self.augment:
                labels = labels[selected_idx]
            labels = self.learning_map_function(labels)
        else:
            labels = np.negative(np.ones(coordinates.shape[0]))

        # REMOVE UNLABELED POINTS IF NOT IN TESTING
        if "label" in self.files.keys():
            labelled_idx = labels != -1
            coordinates = coordinates[labelled_idx]
            features = features[labelled_idx]
            labels = labels[labelled_idx]
            selected_idx = selected_idx[labelled_idx]

        discrete_coords, unique_map = ME.utils.sparse_quantize(
            coordinates=coordinates,
            return_index=True,
            quantization_size=self.voxel_size,
        )

        unique_feats = features[unique_map]
        unique_labels = labels[unique_map]
        selected_idx = selected_idx[unique_map]

        return discrete_coords, unique_feats, unique_labels, selected_idx, t


class SemanticPOSSRestrictedDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            config_file="config/semposs_dataset.yaml",
            split="train",
            voxel_size=0.05,
            downsampling=80000,
            augment=False,
            label_mapping=None,
            data_splits=None,
            superpoint=False
    ):
        self.superpoint = superpoint
        self.data_splits = data_splits
        """Load data from given dataset directory."""

        with open(config_file, "r") as f:
            self.config = yaml.safe_load(f)

        self.files = {"input": []}
        if split != "test":
            self.files["label"] = []
        self.filenames = []

        self.voxel_size = voxel_size
        self.downsampling = downsampling
        self.augment = False
        if split == "train" and augment:
            self.augment = True
            self.clip_bounds = None
            self.scale_augmentation_bound = (0.95, 1.05)
            self.rotation_augmentation_bound = (
                (-np.pi / 20, np.pi / 20),
                (-np.pi / 20, np.pi / 20),
                (-np.pi / 20, np.pi / 20),
            )
            self.translation_augmentation_ratio_bound = None

            self.voxelizer = Voxelizer(
                voxel_size=self.voxel_size,
                clip_bound=self.clip_bounds,
                use_augmentation=self.augment,
                scale_augmentation_bound=self.scale_augmentation_bound,
                rotation_augmentation_bound=self.rotation_augmentation_bound,
                translation_augmentation_ratio_bound=self.translation_augmentation_ratio_bound,
                ignore_label=-1,
            )

        for sequence in self.config["split_sequence"][split]:
            for idx, type in enumerate(self.files.keys()):
                files_path = os.path.join(
                    self.config["dataset_path"],
                    "sequences",
                    sequence,
                    self.config["folder_name"][type],
                )
                if not os.path.exists(files_path):
                    raise RuntimeError("Point cloud directory missing: " + files_path)
                files = os.listdir(files_path)
                data = sorted([os.path.join(files_path, f) for f in files])
                if len(data) == 0:
                    raise RuntimeError("Missing data for " + type)
                self.files[type].extend(data)
                if idx == 0:
                    self.filenames.extend(data)

        learning_map = self.config["learning_map"]
        self.learning_map_function = np.vectorize(lambda x: learning_map[x])

        self.num_files = len(self.filenames)

        if label_mapping is not None:
            self.label_mapping_function = np.vectorize(lambda x: label_mapping[x])
        else:
            self.label_mapping_function = None
        '''
        t = 0
        pc_filename = self.files["input"][0]
        scan = np.fromfile(pc_filename, dtype=np.float32)
        scan = scan.reshape((-1, 4))
        coordinates = scan[:, 0:3]  # get xyz
        remissions = scan[:, 3]  # get remission

        label_filename = self.files["label"][t]
        labels = np.fromfile(label_filename, dtype=np.int32)
        labels = labels.reshape((-1))
        labels = labels & 0xFFFF
        labels = self.learning_map_function(labels)
        coordinates = coordinates[labels != -1]
        mapped_labels = self.label_mapping_function(labels[labels != -1])

        from sklearn import cluster
        clustering = cluster.DBSCAN(eps=0.5, min_samples=2).fit(coordinates[mapped_labels==10])
        region_id = clustering.labels_
        import pickle

        with open("coords_o.pkl", "wb") as f:
            pickle.dump(coordinates[mapped_labels==10], f)
        with open("label.pkl", "wb") as f:
            pickle.dump(region_id, f)
        import ipdb;ipdb.set_trace()
        '''
        if data_splits == 0:
            self.unseen_id = 9
        else:
            self.unseen_id = 10

    def __len__(self):
        return self.num_files

    def __getitem__(self, t):

        pc_filename = self.files["input"][t]
        scan = np.fromfile(pc_filename, dtype=np.float32)
        scan = scan.reshape((-1, 4))
        coordinates = scan[:, 0:3]  # get xyz
        remissions = scan[:, 3]  # get remission
        features = np.ones((coordinates.shape[0], 1))

        # 新增
        import os
        if os.path.exists(
                CACHE_ROOT + "/POSS-update/" + str(self.data_splits) + "/" + str(t) + '.pkl'):
            pass
        else:

            if "label" in self.files.keys():
                label_filename = self.files["label"][t]
                labels = np.fromfile(label_filename, dtype=np.int32)
                labels = labels.reshape((-1))
                labels = labels & 0xFFFF
                for index, element in enumerate(labels):
                    labels[index] = self.config["learning_map"].get(element, -1)
            else:
                labels = np.negative(np.ones(coordinates.shape[0]))

            # REMOVE UNLABELED POINTS IF NOT IN TESTING
            if "label" in self.files.keys():
                labelled_idx = labels != -1
                labels1 = labels[labelled_idx]

            if self.label_mapping_function is not None:
                mapped_labels = self.label_mapping_function(labels1)
            else:
                mapped_labels = np.copy(labels)

            from sklearn import cluster
            if labels1[mapped_labels == self.unseen_id].shape[0] >= 1:
                clustering = cluster.DBSCAN(eps=0.5, min_samples=2).fit(
                    coordinates[labels != -1][mapped_labels == self.unseen_id])
                region_id = clustering.labels_
            else:
                region_id = None
            save_path = CACHE_ROOT + "/POSS-update/" + str(self.data_splits) + "/" + str(t) + '.pkl'
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

            with open(save_path, "wb") as f:
                pickle.dump(region_id, f)

        if self.augment:
            # DOWNSAMPLING
            selected_idx = np.random.choice(
                coordinates.shape[0], self.downsampling, replace=False
            )
            selected_idx = np.sort(selected_idx)
            coordinates = coordinates[selected_idx]
            features = features[selected_idx]

            # TRANSFORMATIONS
            voxel_mtx, affine_mtx = self.voxelizer.get_transformation_matrix()

            rigid_transformation = affine_mtx @ voxel_mtx
            # Apply transformations

            homo_coords = np.hstack(
                (
                    coordinates,
                    np.ones((coordinates.shape[0], 1), dtype=coordinates.dtype),
                )
            )
            # coords = np.floor(homo_coords @ rigid_transformation.T[:, :3])
            coordinates = homo_coords @ rigid_transformation.T[:, :3]
        else:
            selected_idx = np.arange(coordinates.shape[0])

        if "label" in self.files.keys():
            label_filename = self.files["label"][t]
            labels = np.fromfile(label_filename, dtype=np.int32)
            labels = labels.reshape((-1))
            labels = labels & 0xFFFF
        #


            # 新增：
            for index, element in enumerate(labels):
                labels[index] = self.config["learning_map"].get(element, -1)
            sup_label = np.zeros_like(labels) - 1

            if self.superpoint:
                fr = open(CACHE_ROOT + "/POSS-update/" + str(self.data_splits)+"/" + str(t) + '.pkl', "rb")
                superpoint = pickle.load(fr)
                fr.close()
                if superpoint is not None:
                    temp = sup_label[labels != -1]
                    temp[self.label_mapping_function(labels[labels != -1]) == self.unseen_id] = superpoint
                    sup_label[labels != -1] = temp

            if self.augment:
                labels = labels[selected_idx]
                sup_label = sup_label[selected_idx]
        else:
            labels = np.negative(np.ones(coordinates.shape[0]))

        # REMOVE UNLABELED POINTS IF NOT IN TESTING
        if "label" in self.files.keys():

            labelled_idx = labels != -1
            coordinates = coordinates[labelled_idx]
            features = features[labelled_idx]
            labels = labels[labelled_idx]
            sup_label = sup_label[labelled_idx]
            selected_idx = selected_idx[labelled_idx]

            if self.label_mapping_function is not None:
                mapped_labels = self.label_mapping_function(labels)
            else:
                mapped_labels = np.copy(labels)

        discrete_coords, unique_map = ME.utils.sparse_quantize(
            coordinates=coordinates,
            return_index=True,
            quantization_size=self.voxel_size,
        )

        unique_feats = features[unique_map]
        unique_labels = labels[unique_map]
        unique_mapped_labels = mapped_labels[unique_map]
        selected_idx = selected_idx[unique_map]
        unique_sup_label = sup_label[unique_map]
        selected_region_idx = np.unique(unique_sup_label)

        return (
            discrete_coords,
            unique_feats,
            unique_labels,
            selected_idx,
            unique_mapped_labels,
            unique_sup_label,
            selected_region_idx,
            t,
        )


class ArCHRestrictedDataset(torch.utils.data.Dataset):
    """ArCH en bloques .npz (ver experimentacion/arch_entrenamiento/convertir_arch.py).
    Misma salida que SemanticPOSSRestrictedDataset. Cada bloque es una muestra; en entrenamiento se
    submuestrean `downsampling` puntos al azar. Las regiones (superpuntos) de la clase nueva se calculan
    sobre el bloque completo y se guardan en cache. features = unos (solo geometria), como en SemanticPOSS."""

    def __init__(self, config_file="config/arch_dataset.yaml", split="train", voxel_size=0.05, downsampling=60000,
                 augment=False, label_mapping=None, data_splits=None, superpoint=False, max_val_points=150000,
                 dbscan_eps=0.5):
        import glob
        with open(config_file, "r") as f:
            self.config = yaml.safe_load(f)
        raiz = os.environ.get("ARCH_TILES", self.config["dataset_path"])
        self.files = {"input": []}
        for sub in self.config["split_sequence"][split]:
            self.files["input"] += sorted(glob.glob(os.path.join(raiz, sub, "*.npz")))
        if not self.files["input"]:
            raise RuntimeError(f"No hay bloques .npz en {raiz} ({self.config['split_sequence'][split]})")
        self.filenames = self.files["input"]
        self.num_files = len(self.filenames)
        self.voxel_size, self.downsampling, self.split = voxel_size, downsampling, split
        self.max_val_points, self.dbscan_eps = max_val_points, dbscan_eps
        self.superpoint, self.data_splits = superpoint, data_splits
        self.augment = split == "train" and augment
        lm = self.config["learning_map"]
        self.lm_tab = np.full(max(lm) + 1, -1, np.int64)
        for k, v in lm.items():
            self.lm_tab[k] = v
        self.label_mapping = label_mapping
        if label_mapping is not None:
            self.map_tab = np.full(max(label_mapping) + 2, -1, np.int64)
            for k, v in label_mapping.items():
                self.map_tab[k] = v
            self.unseen_id = max(label_mapping.values())   # id de la clase "nueva" (la de los desconocidos)
        if self.augment:
            self.voxelizer = Voxelizer(
                voxel_size=voxel_size, clip_bound=None, use_augmentation=True,
                scale_augmentation_bound=(0.95, 1.05),
                rotation_augmentation_bound=((-np.pi / 20, np.pi / 20),) * 3,
                translation_augmentation_ratio_bound=None, ignore_label=-1)
        self._regiones = {}

    def __len__(self):
        return self.num_files

    def _regiones_bloque(self, t, xyz, lab_map):
        """Region (DBSCAN, eps m) de cada punto de la clase nueva; -1 en el resto. Sobre el bloque completo."""
        if t in self._regiones:
            return self._regiones[t]
        clave = (f"{CACHE_ROOT}/ArCH/{self.data_splits}_{self.unseen_id}_{self.dbscan_eps}/"
                 f"{os.path.basename(self.files['input'][t])}.pkl")
        if os.path.exists(clave):
            with open(clave, "rb") as f:
                reg = pickle.load(f)
        else:
            reg = np.full(len(xyz), -1, np.int64)
            m = lab_map == self.unseen_id
            if m.sum() >= 2:
                from sklearn import cluster
                from scipy.spatial import cKDTree
                p = xyz[m]
                # DBSCAN sobre una version submuestreada (0.2 m) y se propaga al resto: con 5 cm de resolucion
                # DBSCAN directo tendria cientos de vecinos por punto.
                g = np.floor(p / 0.2).astype(np.int64)
                _, idx = np.unique(g, axis=0, return_index=True)
                lab = cluster.DBSCAN(eps=self.dbscan_eps, min_samples=2).fit(p[idx]).labels_
                _, vec = cKDTree(p[idx]).query(p)
                reg[m] = lab[vec]
            os.makedirs(os.path.dirname(clave), exist_ok=True)
            with open(clave, "wb") as f:
                pickle.dump(reg, f)
        self._regiones[t] = reg
        return reg

    def __getitem__(self, t):
        z = np.load(self.files["input"][t])
        coordinates = z["xyz"].astype(np.float32)
        raw = z["label"].astype(np.int64)
        labels = self.lm_tab[np.clip(raw, 0, len(self.lm_tab) - 1)]   # ids aprendidos; -1 = ignorada
        n = len(coordinates)
        if self.superpoint and self.label_mapping is not None:
            lm_full = np.where(labels >= 0, self.map_tab[np.clip(labels, 0, len(self.map_tab) - 1)], -1)
            sup_full = self._regiones_bloque(t, coordinates, lm_full)
        else:
            sup_full = np.full(n, -1, np.int64)

        if self.augment:
            selected_idx = np.sort(np.random.choice(n, min(self.downsampling, n), replace=False))
        elif self.split != "train" and n > self.max_val_points:
            selected_idx = np.sort(np.random.RandomState(t).choice(n, self.max_val_points, replace=False))
        else:
            selected_idx = np.arange(n)
        coordinates, labels, sup_label = coordinates[selected_idx], labels[selected_idx], sup_full[selected_idx]
        features = np.ones((len(coordinates), 1))
        if self.augment:
            voxel_mtx, affine_mtx = self.voxelizer.get_transformation_matrix()
            rigid = affine_mtx @ voxel_mtx
            homo = np.hstack((coordinates, np.ones((len(coordinates), 1), dtype=coordinates.dtype)))
            coordinates = homo @ rigid.T[:, :3]

        ok = labels != -1
        coordinates, features, labels, sup_label, selected_idx = (
            coordinates[ok], features[ok], labels[ok], sup_label[ok], selected_idx[ok])
        mapped_labels = self.map_tab[labels] if self.label_mapping is not None else np.copy(labels)

        discrete_coords, unique_map = ME.utils.sparse_quantize(
            coordinates=coordinates, return_index=True, quantization_size=self.voxel_size)
        unique_sup_label = sup_label[unique_map]
        return (discrete_coords, features[unique_map], labels[unique_map], selected_idx[unique_map],
                mapped_labels[unique_map], unique_sup_label, np.unique(unique_sup_label), t)


class SemanticPOSSRestrictedDatasetCleanSplit(SemanticPOSSRestrictedDataset):
    def __init__(
            self,
            clean_mask,
            config_file="config/semposs_dataset.yaml",
            split="train",
            voxel_size=0.05,
            downsampling=80000,
            augment=False,
            label_mapping=None,
    ):
        super().__init__(
            config_file, split, voxel_size, downsampling, augment, label_mapping
        )
        self.filenames = np.array(self.filenames)[clean_mask]
        self.num_files = len(self.filenames)
        for key in self.files.keys():
            self.files[key] = np.array(self.files[key])[clean_mask]


class dataset_wrapper(torch.utils.data.Dataset):
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, t):
        to_ret = self.dataset.__getitem__(t)[:-1] + self.dataset.__getitem__(t)

        return to_ret