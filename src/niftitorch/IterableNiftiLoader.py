from concurrent.futures import ThreadPoolExecutor
import random

from torch.utils.data import IterableDataset
from scipy.ndimage import zoom
import nibabel as nib
from tqdm import tqdm
import numpy as np

from .utils import get_matched_ids, get_filepath_list_from_id


class IterableNiftiDataset(IterableDataset):
    '''
    Iterable Dataset class for Nifti data.
    This class takes as input two matching datasets of Nifti images
    and returns one the data in one of the following ways:
        - 2d slices of the images across any axis
        - 2d slices with a width of n slices on either side of the slice
    This class also implements psuedo random sampling of the data by
    using a buffer of scans from which images are sampled in random order.
    once the buffer is empty, it is refilled with new scans. This significantly
    increase speed of data loading due to the fact that scans are read
    sequentially
    from disk.

    Parameters
    ----------
    input_dir : str
        Path to the directory containing the input nifti files
    mask_dir : str
        Path to the directory containing the mask nifti files
    transform : callable
        Implementation of a torchvision transform to apply to the data
    split_char : str, optional
        Character used to split the UID from the filenames of the matched
        files. an example file might look like this 1234-t1.nii.gz, where
        1234 is the UID.
        default is "-"
    scan_size : str or tuple, optional
        Size to resample all images to before loading. Options are 'most',
        'largest', 'smallest', 'false', or a tuple of the desired shape.
        'most' resamples all images to the most common shape, 'largest'
        resamples all images to the largest shape, 'smallest' resamples all
        images to the smallest shape, and 'false' does not resample the images.
        default is 'most'.
        NOTE: While resampling on the fly is supported, it does incur a
        signifcant performance cost. If dataloading becomes a bottleneck,
        it is recommended to preprocess your dataset to a uniform shape.
    slice_axis : int, optional
        This argument is used if out_type is 'slice' or 'width_slice'. It
        specifies the axis to slice the images along. Options are 0, 1, 2, or
        'all'. 'all' will return all slices of the image along all axes. All
        requires that the images are Isotropic.
        default is 2 (z-axis)
    slice_width : int, optional
        This argument is used if out_type is 'width_slice'. It specifies the
        width of the slices to return. Must be an odd number.
        default is 1
    width_labels : bool, optional
        This argument is used if out_type is 'width_slice'. If True, the
        labels will be returned as a 3d volume with the same width as the
        slices. If False, the labels will be returned as a 2d slice
        corresponding to the center slice of the input.
        default is False
    buffer_size : int, optional
        Number of scans to store in buffer, Increasing this will increase
        the amound of randomness in the sampling.
        default is 2
    '''
    def __init__(self, input_dir, mask_dir, transform,
                 split_char="-", scan_size='most',
                 slice_width=1, width_labels=False,
                 buffer_size=2, slice_axis=2):
        self.input_dir = input_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.split_char = split_char
        self.scan_size = scan_size
        self.slice_width = slice_width
        self.width_labels = width_labels
        self.buffer_size = buffer_size
        self.slice_axis = slice_axis

        self.shape_frequencies = {}
        self.scan_list = self._load_scan_list()
        self.used_scans = []
        self.scan_buffer = []
        self.buffer_indexs = []

    def _get_num_slices(self, scan):
        '''
        Returns the number of slices for the input image input
        '''
        # Check if scan is a filepath (string), then load it; otherwise,
        # use it directly
        if isinstance(scan, str):
            scan_data = nib.load(scan, mmap=True)
        else:
            scan_data = scan

        # Handle the 'all' case or a specific axis
        if self.slice_axis == "all":
            shape = scan_data.shape
            return sum(shape)
        else:
            return scan_data.shape[self.slice_axis]

    def _get_resample_shape(self):
        scan_size = self.scan_size
        if scan_size == 'most':
            new_shape = max(self.shape_frequencies,
                            key=self.shape_frequencies.get)

        elif scan_size == 'largest':
            largest_shape = None
            largest_size = 0
            for shape, occurrences in self.shape_frequencies.items():
                size = shape[0] * shape[1] * shape[2]
                if size > largest_size:
                    largest_shape = shape
                    largest_size = size
            new_shape = largest_shape

        elif scan_size == 'smallest':
            smallest_shape = None
            smallest_size = np.inf
            for shape, occurrences in self.shape_frequencies.items():
                size = shape[0] * shape[1] * shape[2]
                if size < smallest_size:
                    smallest_shape = shape
                    smallest_size = size
            new_shape = smallest_shape

        else:
            new_shape = scan_size

        return new_shape

    def _resample_image(self, image, new_shape):
        '''
        Resamples the input image to the new shape
        '''

        zoom_factors = [new_dim / old_dim for new_dim, old_dim in
                        zip(new_shape, image.shape)]
        reasampled_image = zoom(image, zoom_factors, order=1)
        return reasampled_image

    def _load_scan_list(self):
        ids = get_matched_ids([self.input_dir, self.mask_dir],
                              split_char=self.split_char)

        # mutlthreading starts here
        with ThreadPoolExecutor() as executor:
            result_list = list(tqdm(executor.map(self._load_scan, ids),
                                    total=len(ids)))

        # result list is [{'input': input_scan, 'mask': mask_scan,
        #                  'slices': slices}, ...]
        return result_list

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

            self._update_shape_frequencies(input_scan.shape)

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

    def _update_shape_frequencies(self, shape):
        if shape not in self.shape_frequencies:
            self.shape_frequencies[shape] = 1
        else:
            self.shape_frequencies[shape] += 1

    def _fill_buffer(self):
        scans = []
        while len(scans) < self.buffer_size:
            random_index = np.random.randint(len(self.scan_list))
            scan = self.scan_list.pop(random_index)
            scans.append(scan)
            self.used_scans.append(scan)
            if len(self.scan_list) == 0:
                break

        buffer_list = []
        slices = 0
        if self.slice_width == 1:
            padding = 0
        else:
            padding = self.slice_width // 2

        for i, scan in enumerate(scans):
            first_index = slices
            slices += scan.get('slices') - (padding*2)
            last_index = slices - 1
            image_data = scan.get('input').get_fdata()
            mask_data = scan.get('mask').get_fdata()

            if self.scan_size != self._get_resample_shape():
                new_shape = self._get_resample_shape()
                image_data = self._resample_image(image_data, new_shape)
                mask_data = self._resample_image(mask_data, new_shape)

            buffer_list.append({'input': image_data, 'mask': mask_data,
                                'first': first_index, 'last': last_index})

        self.scan_buffer = buffer_list
        self.buffer_indexs = random.sample(list(range(slices)), slices)

    def _load_slice(self, idx):
        scan_to_use = None
        for scan in self.scan_buffer:
            if scan.get('first') <= idx <= scan.get('last'):
                scan_to_use = scan
                break

        if scan_to_use is None:
            raise ValueError("Index not found in buffer")

        slice_index = idx - scan_to_use.get('first')

        return self._get_slices(scan_to_use, slice_index)

    def _get_slices(self, scan, slice_idx):
        if self.slice_width == 1:
            offset = 0
        else:
            offset = self.slice_width // 2
        slice_idx += offset
        start_idx = slice_idx - offset
        end_idx = slice_idx + offset + 1

        # Determine the slicing based on the axis
        slice_none = slice(None, None, None)  # Equivalent to ':'
        slices_input = [slice_none, slice_none, slice_none]
        slices_mask = [slice_none, slice_none, slice_none]

        if self.width_labels:
            slices_input[self.slice_axis] = slice(start_idx, end_idx)
            slices_mask[self.slice_axis] = slice(start_idx, end_idx)
        else:
            slices_input[self.slice_axis] = slice(start_idx, end_idx)
            slices_mask[self.slice_axis] = slice(slice_idx, slice_idx + 1)

        input_slice = scan.get("input")[tuple(slices_input)]
        mask_slice = scan.get("mask")[tuple(slices_mask)]

        return self.transform(input_slice, mask_slice)

    def __iter__(self):
        return self

    def __next__(self):
        if len(self.buffer_indexs) == 0:
            if len(self.scan_list) == 0:
                self.scan_list = self.used_scans
                self.used_scans = []
                raise StopIteration
            else:
                self._fill_buffer()

        idx = self.buffer_indexs.pop()
        return self._load_slice(idx)
