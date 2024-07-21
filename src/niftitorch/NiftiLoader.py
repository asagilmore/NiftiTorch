from concurrent.futures import ThreadPoolExecutor

from torch.utils.data import Dataset
import nibabel as nib
from tqdm import tqdm
import numpy as np
from scipy.ndimage import zoom
from .utils import get_matched_ids, get_filepath_list_from_id


class NiftiDataset(Dataset):
    """
    Dataset class for Nifti to Nifti data.
    This class takes as input two matching datasets of Nifti images
    and returns one the data in one of the following ways:
    - 2d slices of the images across any axis
    - 2d slices with a width of n slices on either side of the slice

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
    """
    def __init__(self, input_dir, mask_dir, transform,
                 split_char="-", scan_size='most',
                 slice_axis=2, slice_width=1, width_labels=False,
                 force_no_resample=False):
        for name, value in locals().items():
            if name != "self":
                setattr(self, name, value)

        if self.slice_width % 2 == 0:
            raise ValueError("Slice width must be an odd number")

        self.shape_frequencies = {}

        self.scan_list = self._load_scan_list()

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

        # Set the appropriate slice based on the axis
        if self.width_labels:
            slices_input[self.slice_axis] = slice(start_idx, end_idx)
            slices_mask[self.slice_axis] = slice(start_idx, end_idx)
        else:
            slices_input[self.slice_axis] = slice(start_idx, end_idx)
            slices_mask[self.slice_axis] = slice(slice_idx, slice_idx + 1)

        input_slice = scan_object.get("input").dataobj[tuple(slices_input)]
        mask_slice = scan_object.get("mask").dataobj[tuple(slices_mask)]

        if len(self.shape_frequencies) == 1 or self.force_no_resample:
            return self.transform(input_slice.copy(), mask_slice.copy())
        else:
            image_resample_shape = list(self._get_resample_shape())
            mask_resample_shape = list(self._get_resample_shape())
            if self.width_labels:
                image_resample_shape[self.slice_axis] = self.slice_width
                mask_resample_shape[self.slice_axis] = self.slice_width
            else:
                image_resample_shape[self.slice_axis] = self.slice_width
                mask_resample_shape[self.slice_axis] = 1

            if input_slice.shape != tuple(image_resample_shape):
                input_slice = self._resample_image(input_slice,
                                                   image_resample_shape)
            if mask_slice.shape != tuple(mask_resample_shape):
                mask_slice = self._resample_image(mask_slice,
                                                  mask_resample_shape)

            return self.transform(input_slice.copy(), mask_slice.copy())

    def _update_shape_frequencies(self, shape):
        if shape not in self.shape_frequencies:
            self.shape_frequencies[shape] = 1
        else:
            self.shape_frequencies[shape] += 1

    def _resample_image(self, image, new_shape):
        """
        Resamples the input image to the new shape
        """

        zoom_factors = [new_dim / old_dim for new_dim, old_dim in
                        zip(new_shape, image.shape)]
        reasampled_image = zoom(image, zoom_factors, order=1)
        return reasampled_image

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

    def _load_scan(self, id):
        input_paths = get_filepath_list_from_id(self.input_dir, id)
        mask_paths = get_filepath_list_from_id(self.mask_dir, id)

        if len(input_paths) == 1 and len(mask_paths) == 1:

            input_scan = nib.load(input_paths[0], mmap=True)
            mask_scan = nib.load(mask_paths[0], mmap=True)

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
                              'first_index': first_index})

        return scan_list

    def _get_num_slices(self, scan):
        """
        Returns the number of slices for the input image input
        """
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


class NiftiDataset3d(NiftiDataset):
    def __init__(self, input_dir, mask_dir, transform, volume_shape=None,
                 *args, **kwargs):
        self.volume_shape = volume_shape
        super().__init__(input_dir, mask_dir, transform, *args, **kwargs)

    def __len__(self):
        return len(self.scan_list)

    def __getitem__(self, idx):
        # handle negative indexing
        if idx < 0:
            idx = len(self) + idx

        image = self.scan_list[idx].get("input")
        mask = self.scan_list[idx].get("mask")

        image = image.get_fdata()
        mask = mask.get_fdata()

        if self.volume_shape:
            image, mask = self._get_volume(image, mask)
            return self.transform(image, mask)
        else:
            return self.transform(image, mask)

    def _get_volume(self, image, mask):
        out_shape = self.volume_shape
        x_index = np.random.randint(0,
                                    image.shape[0] - out_shape[0] + 1)
        y_index = np.random.randint(0,
                                    image.shape[1] - out_shape[1] + 1)
        z_index = np.random.randint(0,
                                    image.shape[2] - out_shape[2] + 1)

        image = image[x_index:x_index + out_shape[0],
                      y_index:y_index + out_shape[1],
                      z_index:z_index + out_shape[2]]

        mask = mask[x_index:x_index + out_shape[0],
                    y_index:y_index + out_shape[1],
                    z_index:z_index + out_shape[2]]

        return image, mask
