from concurrent.futures import ThreadPoolExecutor

import torch
from torch.utils.data import IterableDataset
from .NiftiDataset import NiftiDataset
import nibabel as nib
from tqdm import tqdm
import numpy as np

from .utils import get_matched_ids, get_filepath_list_from_id


class IterableNiftiDataset(IterableDataset):
    def __init__(self, input_dir, mask_dir, transform,
                 split_char="-", slice_width=1, width_labels=False):
        self.input_dir = input_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.split_char = split_char
        self.slice_width = slice_width
        self.width_labels = width_labels

        self.scan_list = self._load_scan_list()
        self.scan_buffer = []


    def _load_scan_list(self):
        ids = get_matched_ids([self.input_dir, self.mask_dir],
                              split_char=self.split_char)

        # result_list = []
        # for id in tqdm(ids):
        #     result_list.append(self._load_scan(id))

        # mutlthreading starts here
        with ThreadPoolExecutor() as executor:
            result_list = list(tqdm(executor.map(self._load_scan, ids),
                                    total=len(ids)))

        # now we count up the slices and add the first and last index
        scan_list = []
        slices = 0
        # padding to save on either side for slice width
        if self.slice_width == 1:
            padding = 0
        else:
            padding = self.slice_width // 2

        for i, result in enumerate(result_list):
            first_index = slices
            slices += result.get('slices') - (padding*2)
            last_index = slices - 1
            scan_list.append({'input': result.get('input'),
                              'mask': result.get('mask'),
                              'last_index': last_index,
                              'first_index': first_index,
                              'used': False
                              })

        return scan_list

    def _load_scan(self, id):
        input_paths = get_filepath_list_from_id(self.input_dir, id)
        mask_paths = get_filepath_list_from_id(self.mask_dir, id)

        if len(input_paths) == 1 and len(mask_paths) == 1:

            input_scan = nib.load(input_paths[0], mmap=True)
            mask_scan = nib.load(mask_paths[0], mmap=True)

            if input_scan.shape != mask_scan.shape:

                raise ValueError(f"ID {id} has Input shape {input_scan.shape} "
                                 f"and Mask shape {mask_scan.shape}. They "
                                 "should be equal.")

            input_slices = self._get_num_slices(input_scan)
            mask_slices = self._get_num_slices(mask_scan)

            if input_slices != mask_slices:
                raise ValueError(f"ID {id} has {input_slices} input slices "
                                 f"and {mask_slices} mask slices. They "
                                 "should be equal.")
            else:
                slices = input_slices

        else:
            raise ValueError("Multiple scans found for ID "
                             "there should be only one Input and Mask for "
                             f"{id}")

        return {'input': input_scan, 'mask': mask_scan, 'slices': slices}

    def __iter__(self):

