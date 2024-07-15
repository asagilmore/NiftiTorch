from concurrent.futures import ThreadPoolExecutor

from torch.utils.data import Dataset
import nibabel as nib
from tqdm import tqdm
import numpy as np
from scipy.ndimage import zoom
from .utils import get_matched_ids, get_filepath_list_from_id


class NiftiDataset(Dataset):
    '''
    Dataset class for Nifti to Nifti data.
    This class takes as input two matching datasets of Nifti images
    and returns one the data in any of the following ways:
        - 2d slices of the images across any axis
        - 2d slices with a width of n slices on either side of the slice
        - 3d volumes of the entire image
        - 3d volumes of a specified shape, sampled randomly from the images


    Parameters
    ----------
    input_dir : str
        Path to the directory containing the input nifti files
    mask_dir : str
        Path to the directory containing the mask nifti files
    transform : callable
        Implementation of a torchvision transform to apply to the data
    out_type : str, optional
        Type of data to return. Options are 'slice', 'width_slice', 'volume'.
        'slice' causes the dataset to output 2d slices from the nifti files,
        'width_slice' returns slices with a width of n stacked along the
        channel dimension, and 'volume' returns a 3d volume, either the entire
        volume or of a specified shape.
        default is "slice"
    split_char : str, optional
        Character used to split the UID from the filenames of the matched
        files. an example file might look like this 1234-t1.nii.gz, where
        1234 is the UID.
        default is "-"
    preload_dtype : str, optional
        Data type to load the nifti files as. "Float16" can be used to save
        memory, but may reduce precision.
        default is "float32".
    scan_size : str or tuple, optional
        Size to resample all images to before loading. Options are 'most',
        'largest', 'smallest', 'false', or a tuple of the desired shape.
        'most' resamples all images to the most common shape, 'largest'
        resamples all images to the largest shape, 'smallest' resamples all
        images to the smallest shape, and 'false' does not resample the images.
        default is 'most'
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
    volume_shape : tuple, optional
        This argument is used if out_type is 'volume'. It specifies the shape
        of the volume to return. If None, the entire volume is returned.
        default is None
    mmap : bool, optional
        This argument is used to load the nifti files as memory-mapped files.
        This is useful if the dataset cannot fit in memory, but signifiticaly
        increases batch loading time. If used it is recoomented to use multiple
        workers with a high prefetch factor.
    '''
    def __init__(self, input_dir, mask_dir, transform, out_type="slice",
                 split_char="-", preload_dtype="float32", scan_size='most',
                 slice_axis=2, slice_width=1, width_labels=False,
                 volume_shape=None, mmap=False):
        for name, value in locals().items():
            if name != "self":
                setattr(self, name, value)

        if self.slice_width % 2 == 0:
            raise ValueError("Slice width must be an odd number")

        self.shape_frequencies = {}

        self.scan_list = self._load_scan_list()
        if not self.mmap:
            self._resample_scan_list(scan_size)

    def __len__(self):
        if self.scan_list:
            return self.scan_list[-1].get("last_index") + 1
        else:
            return 0

    def __getitem__(self, idx):
        # handle negative indexing
        if idx < 0:
            idx = len(self) + idx

        scan_to_use = None
        for scan in self.scan_list:
            if scan.get("first_index") <= idx <= scan.get("last_index"):
                scan_to_use = scan
                break

        if scan_to_use is None:
            raise IndexError(f"Index {idx} out of range for dataset")

        slice_index = idx - scan_to_use.get("first_index")

        return self._get_slices(scan_to_use, slice_index)

    def _get_slices(self, scan_object, slice_idx):
        if self.mmap:
            input_scan = scan_object.get("input").get_fdata()
            mask_scan = scan_object.get("mask").get_fdata()
            new_shape = self._get_resample_shape()
            if input_scan.shape != new_shape:
                input_scan = self._resample_image(input_scan, new_shape)
            if mask_scan.shape != new_shape:
                mask_scan = self._resample_image(mask_scan, new_shape)
        else:
            input_scan = scan_object.get("input")
            mask_scan = scan_object.get("mask")

        # because first and last indexs are set to not include padding
        # we need to add the padding back to the index
        if self.slice_width == 1:
            offset = 0
        else:
            offset = self.slice_width // 2
        slice_idx += offset
        start_idx = slice_idx - offset
        # end_idx is exclusive so add 1
        end_idx = slice_idx + offset + 1

        if self.slice_axis == 0:
            input_slice = input_scan[start_idx:end_idx, :, :]
            if self.width_labels:
                mask_slice = mask_scan[start_idx:end_idx, :, :]
            else:
                mask_slice = mask_scan[slice_idx, :, :]
            mask_slice = mask_scan[start_idx:end_idx, :, :]
        elif self.slice_axis == 1:
            input_slice = input_scan[:, start_idx:end_idx, :]
            if self.width_labels:
                mask_slice = mask_scan[:, start_idx:end_idx, :]
            else:
                mask_slice = mask_scan[:, slice_idx, :]
        elif self.slice_axis == 2:
            input_slice = input_scan[:, :, start_idx:end_idx]
            if self.width_labels:
                mask_slice = mask_scan[:, :, start_idx:end_idx]
            else:
                mask_slice = mask_scan[:, :, slice_idx]

        input_slice, mask_slice = self.transform(input_slice, mask_slice)

        return input_slice, mask_slice

    def _update_shape_frequencies(self, shape):
        if shape not in self.shape_frequencies:
            self.shape_frequencies[shape] = 1
        else:
            self.shape_frequencies[shape] += 1

    def _resample_image(self, image, new_shape):
        '''
        Resamples the input image to the new shape
        '''
        image = image.astype(np.float32)
        zoom_factors = [new_dim / old_dim for new_dim, old_dim in
                        zip(new_shape, image.shape)]
        reasampled_image = zoom(image, zoom_factors, order=1)
        return reasampled_image.astype(self.preload_dtype)

    def _get_resample_shape(self):
        scan_size = self.scan_size
        if scan_size == 'most':
            new_shape = max(self.shape_frequencies,
                            key=self.shape_frequencies.get)

        elif scan_size == 'largest':
            largest_shape = None
            largest_size = 0
            for shape, occurrences in self.shapes_dict.items():

                size = shape[0] * shape[1] * shape[2]
                if size > largest_size:
                    largest_shape = shape
                    largest_size = size
            new_shape = largest_shape

        elif scan_size == 'smallest':
            smallest_shape = None
            smallest_size = np.inf
            for shape, occurrences in self.shapes_dict.items():
                size = shape[0] * shape[1] * shape[2]
                if size < smallest_size:
                    smallest_shape = shape
                    smallest_size = size
            new_shape = smallest_shape

        else:
            new_shape = scan_size
        return new_shape

    def _resample_scan_list(self, scan_size):
        new_shape = self._get_resample_shape()
        for scan in self.scan_list:
            if scan['input'].shape != new_shape:
                scan['input'] = self._resample_image(scan['input'], new_shape)
            if scan['mask'].shape != new_shape:
                scan['mask'] = self._resample_image(scan['mask'], new_shape)

    def _load_scan(self, id):
        input_paths = get_filepath_list_from_id(self.input_dir, id)
        mask_paths = get_filepath_list_from_id(self.mask_dir, id)

        if len(input_paths) == 1 and len(mask_paths) == 1:

            if self.mmap:
                input_scan = nib.load(input_paths[0], mmap=True)
                mask_scan = nib.load(mask_paths[0], mmap=True)
            else:
                input_scan = nib.load(input_paths[0]).get_fdata()
                mask_scan = nib.load(mask_paths[0]).get_fdata()

            # update shape frequencies
            if input_scan.shape != mask_scan.shape:

                raise ValueError(f"ID {id} has Input shape {input_scan.shape} "
                                 f"and Mask shape {mask_scan.shape}. They "
                                 "should be equal.")
            else:
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

    def _load_scan_list(self):
        ids = get_matched_ids([self.input_dir, self.mask_dir],
                              split_char=self.split_char)

        # mutlthreading starts here
        with ThreadPoolExecutor() as executor:
            result_list = list(tqdm(executor.map(self._load_scan, ids),
                                    total=len(ids)))
            # we now have a list as follows:
            # [{'input': input_scan, 'mask': mask_scan, 'slices': slices}, ...]

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
                              'first_index': first_index})

        return scan_list

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
