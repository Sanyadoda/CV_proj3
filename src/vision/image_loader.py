"""
Script with Pytorch's dataloader class
"""

import glob
import os
from typing import Dict, List, Tuple

import torch
import torch.utils.data as data
import torchvision
from PIL import Image
import csv
import pandas as pd


class ImageLoader(data.Dataset):
    """Class for data loading"""

    train_folder = "train"
    test_folder = "test"

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        transform: torchvision.transforms.Compose = None,
    ):
        """Initialize the dataloader and set `curr_folder` for the corresponding data split.

        Args:
            root_dir: the dir path which contains the train and test folder
            split: 'test' or 'train' split
            transform: the composed transforms to be applied to the data
        """
        self.root = os.path.expanduser(root_dir)
        self.transform = transform
        self.split = split

        if split == "train":
            self.curr_folder = os.path.join(root_dir, self.train_folder)
        elif split == "test":
            self.curr_folder = os.path.join(root_dir, self.test_folder)

        self.class_dict = self.get_classes()
        self.dataset = self.load_imagepaths_with_labels(self.class_dict)

    def load_imagepaths_with_labels(
        self, class_labels: Dict[str, int]
    ) -> List[Tuple[str, int]]:
        """Fetches all (image path,label) pairs in the dataset.

        Args:
            class_labels: the class labels dictionary, with keys being the classes in this dataset
        Returns:
            list[(filepath, int)]: a list of filepaths and their class indices
        """

        img_paths = []  # a list of (filename, class index)

        ############################################################################
        # Student code begin
        ############################################################################

        for cls, idx in class_labels.items():
            cls_path = os.path.join(self.curr_folder, cls)
            for img_file in os.listdir(cls_path):
                if img_file.endswith(('.jpg', '.png', '.jpeg')):  # Ensure valid image format
                    img_paths.append((os.path.join(cls_path, img_file), idx))

        """ raise NotImplementedError(
            "`load_imagepaths_with_labels` function in "
            + "`image_loader.py` needs to be implemented"
        )"""

        ############################################################################
        # Student code end
        ############################################################################

        return img_paths

    def get_classes(self) -> Dict[str, int]:
        """Get the classes (which are folder names in self.curr_folder)

        NOTE: Please make sure that your classes are sorted in alphabetical order
        i.e. if your classes are ['apple', 'giraffe', 'elephant', 'cat'], the
        class labels dictionary should be:
        {"apple": 0, "cat": 1, "elephant": 2, "giraffe":3}

        If you fail to do so, you will most likely fail the accuracy
        tests on Gradescope

        Returns:
            Dict of class names (string) to integer labels
        """

        classes = dict()
        ############################################################################
        # Student code begin
        ############################################################################

        #Sort alphabetically
        classes = sorted(os.listdir(self.curr_folder))
        #Assign int value to each class
        classes = {cls: idx for idx, cls in enumerate(classes)}

        """ raise NotImplementedError(
            "`get_classes` function in "
            + "`image_loader.py` needs to be implemented"
        ) """

        ############################################################################
        # Student code end
        ############################################################################
        return classes

    def load_img_from_path(self, path: str) -> Image:
        """Loads an image as grayscale (using Pillow).

        Note: do not normalize the image to [0,1]

        Args:
            path: the file path to where the image is located on disk
        Returns:
            image: grayscale image with values in [0,255] loaded using pillow
                Note: Use 'L' flag while converting using Pillow's function.
        """

        img = None
        ############################################################################
        # Student code begin
        ############################################################################

        img=Image.open(path).convert('L')

        """ raise NotImplementedError(
            "`load_img_from_path` function in "
            + "`image_loader.py` needs to be implemented"
        ) """

        ############################################################################
        # Student code end
        ############################################################################
        return img

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        """Fetches the item (image, label) at a given index.

        Note: Do not forget to apply the transforms, if they exist

        Hint:
        1) get info from self.dataset
        2) use load_img_from_path
        3) apply transforms if valid

        Args:
            index: Index
        Returns:
            img: image of shape (H,W)
            class_idx: index of the ground truth class for this image
        """
        img = None
        class_idx = None

        ############################################################################
        # Student code start
        ############################################################################

        img_path, class_idx = self.dataset[index]
        img = self.load_img_from_path(img_path)

        if self.transform:
            img = self.transform(img)

        """ raise NotImplementedError(
            "`__getitem__` function in "
            + "`image_loader.py` needs to be implemented"
        ) """

        ############################################################################
        # Student code end
        ############################################################################
        return img, class_idx

    def __len__(self) -> int:
        """Returns the number of items in the dataset.

        Returns:
            l: length of the dataset
        """

        l = 0

        ############################################################################
        # Student code start
        ############################################################################

        l=len(self.dataset)
        """ raise NotImplementedError(
            "`__len__` function in "
            + "`image_loader.py` needs to be implemented"
        ) """
        ############################################################################
        # Student code end
        ############################################################################
        return l


class MultiLabelImageLoader(data.Dataset):
    """Class for data loading"""

    train_folder = "train"
    test_folder = "test"

    def __init__(
        self,
        root_dir: str,
        labels_csv: str,
        split: str = "train",
        transform: torchvision.transforms.Compose = None,
    ):
        """Initialize the dataloader and set `curr_folder` for the corresponding data split.

        Args:
            root_dir: the dir path which contains the train and test folder
            labels_csv: the path to the csv file containing ground truth labels
            split: 'test' or 'train' split
            transform: the composed transforms to be applied to the data
        """
        self.root = os.path.expanduser(root_dir)
        self.labels_csv = labels_csv
        self.transform = transform
        self.split = split

        if split == "train":
            self.curr_folder = os.path.join(root_dir, self.train_folder)
        elif split == "test":
            self.curr_folder = os.path.join(root_dir, self.test_folder)

        self.dataset = self.load_imagepaths_with_labels()

    def load_imagepaths_with_labels(self) -> List[Tuple[str, torch.Tensor]]:
        """Fetches all (image path,labels) pairs in the dataset from csv file. Ensure that only
        the images from the classes in ['coast', 'highway', 'mountain', 'opencountry', 'street']
        are included. 

        NOTE: Be mindful of the returned labels type

        Returns:
            list[(filepath, list(int))]: a list of filepaths and their labels
        """

        img_paths = []  # a list of (filename, class index)

        ############################################################################
        # Student code begin
        ############################################################################
        
        # Define valid classes
        valid_classes = {'coast', 'highway', 'mountain', 'opencountry', 'street'}

        # Read CSV file
        df = pd.read_csv(self.labels_csv, header=None)

        # Iterate over each row
        for _, row in df.iterrows():
            class_name, filename, *labels = row  # Unpack row values

            if class_name in valid_classes:
                full_path = os.path.join(self.curr_folder, class_name, filename)  # Construct full image path
                labels_tensor = torch.tensor(labels, dtype=torch.float32)  # Convert labels to tensor
                img_paths.append((full_path, labels_tensor))


        """ raise NotImplementedError(
            "`load_imagepaths_with_labels` function in "
            + "`image_loader.py` needs to be implemented"
        ) """

        ############################################################################
        # Student code end
        ############################################################################

        return img_paths


    def load_img_from_path(self, path: str) -> Image:
        """Loads an image as grayscale (using Pillow).

        Note: do not normalize the image to [0,1]

        Args:
            path: the file path to where the image is located on disk
        Returns:
            image: grayscale image with values in [0,255] loaded using pillow
                Note: Use 'L' flag while converting using Pillow's function.
        """

        img = None
        ############################################################################
        # Student code begin
        ############################################################################

        img=Image.open(path).convert('L')

        """ raise NotImplementedError(
            "`load_img_from_path` function in "
            + "`image_loader.py` needs to be implemented"
        ) """


        ############################################################################
        # Student code end
        ############################################################################
        return img

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Fetches the item (image, label) at a given index.

        Note: Do not forget to apply the transforms, if they exist

        Hint:
        1) get info from self.dataset
        2) use load_img_from_path
        3) apply transforms if valid

        Args:
            index: Index
        Returns:
            img: image of shape (H,W)
            class_idxs: indices of shape (num_classes, ) of the ground truth classes for this image
        """
        img = None
        class_idxs = None

        ############################################################################
        # Student code start
        ############################################################################
        
        img_path, class_idxs = self.dataset[index]
        img = self.load_img_from_path(img_path)

        if self.transform:
            img = self.transform(img)
        
        """ raise NotImplementedError(
            "`__getitem__` function in "
            + "`image_loader.py` needs to be implemented"
        ) """

        ############################################################################
        # Student code end
        ############################################################################
        return img, class_idxs

    def __len__(self) -> int:
        """Returns the number of items in the dataset.

        Returns:
            l: length of the dataset
        """

        l = 0

        ############################################################################
        # Student code start
        ############################################################################
        
        l=len(self.dataset)
        """ raise NotImplementedError(
            "`__len__` function in "
            + "`image_loader.py` needs to be implemented"
        ) """
        ############################################################################
        # Student code end
        ############################################################################
        return l
