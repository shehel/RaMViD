from random import sample
from PIL import Image, ImageSequence
import blobfile as bf
from mpi4py import MPI
import numpy as np
from torch.utils.data import DataLoader, Dataset
import torch
import av
import pdb
from typing import Any
from typing import Callable
from typing import Optional
from typing import Tuple
from typing import Union
import h5py
from pathlib import Path


def load_data(
    *, data_dir, batch_size, image_size, class_cond=False, deterministic=False, rgb=True, seq_len=20
):
    """
    For a dataset, create a generator over (videos, kwargs) pairs.

    Each video is an NCLHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensors of class labels.

    :param data_dir: a dataset directory.
    :param batch_size: the batch size of each returned pair.
    :param image_size: the size to which frames are resized.
    :param class_cond: if True, include a "y" key in returned dicts for class
                       label. If classes are not available and this is true, an
                       exception will be raised.
    :param deterministic: if True, yield results in a deterministic order.
    """
    if not data_dir:
        raise ValueError("unspecified data directory")
    all_files = _list_video_files_recursively(data_dir)
    classes = None
    if class_cond:
        # Assume classes are the first part of the filename,
        # before an underscore.
        class_names = [bf.basename(path).split("_")[0] for path in all_files]
        sorted_classes = {x: i for i, x in enumerate(sorted(set(class_names)))}
        classes = [sorted_classes[x] for x in class_names]
    # if entry is empty


    # TODO Fix the logic here as mp4 inside data will lead to .h5 branch not being executed
    if all_files == []:
        file_filter = "**/training/*8ch.h5"
        all_files = list(Path(data_dir).rglob(file_filter))
        all_files.sort()

    entry = str(all_files[0]).split(".")[-1]
    collate_fn = None

    if entry in ["avi", "mp4"]:
        dataset = VideoDataset_mp4(
            image_size,
            all_files,
            classes=classes,
            shard=MPI.COMM_WORLD.Get_rank(),
            num_shards=MPI.COMM_WORLD.Get_size(),
            rgb=rgb,
            seq_len=seq_len
        )
    elif entry == "h5":
        dataset = T4C_dataset(
            image_size,
            all_files,
            shard=MPI.COMM_WORLD.Get_rank(),
            num_shards=MPI.COMM_WORLD.Get_size(),
            seq_len=seq_len
        )
        collate_fn = train_collate_fn
    elif entry in ["gif"]:
        dataset = VideoDataset_gif(
            image_size,
            all_files,
            classes=classes,
            shard=MPI.COMM_WORLD.Get_rank(),
            num_shards=MPI.COMM_WORLD.Get_size(),
            rgb=rgb,
            seq_len=seq_len
        )
    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True,
            collate_fn=collate_fn
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True,
            collate_fn=collate_fn
        )
    while True:
        yield from loader

def train_collate_fn(batch):
    dynamic_input_batch = batch
    dynamic_input_batch = np.stack(dynamic_input_batch, axis=0)
    #target_batch = np.stack(target_batch, axis=0)
    dynamic_input_batch = np.moveaxis(dynamic_input_batch, source=4, destination=1)
    dynamic_input_batch = torch.from_numpy(dynamic_input_batch).float()
    #target_batch = np.moveaxis(target_batch, source=4, destination=1)
    #target_batch = torch.from_numpy(target_batch).float()
    dynamic_input_batch = dynamic_input_batch.reshape(-1, 48, 128, 128)
    #target_batch = target_batch.reshape(-1, channels, self.h, self.w)

    #target_batch = F.pad(target_batch, pad=self.pad_tuple)
    #dynamic_input_batch = F.pad(dynamic_input_batch, pad=self.pad_tuple)
    #target_batch = torch.divide(target_batch, 255.0) # bring the upper range to 1
    #dynamic_input_batch = torch.divide(dynamic_input_batch, 255.0) # bring the upper range to 1

    #target_batch = 2 * target_batch - 1
    #dynamic_input_batch = 2 * dynamic_input_batch - 1


    return dynamic_input_batch, {}



MAX_TEST_SLOT_INDEX = 240

# helpers functions
def load_h5_file(file_path: Union[str, Path], sl: Optional[slice] = None, to_torch: bool = False) -> np.ndarray:
    """Given a file path to an h5 file assumed to house a tensor, load that
    tensor into memory and return a pointer.
    Parameters
    ----------
    file_path: str
        h5 file to load
    sl: Optional[slice]
        slice to load (data is written in chunks for faster access to rows).
    """
    # load
    with h5py.File(str(file_path) if isinstance(file_path, Path) else file_path, "r") as fr:
        data = fr.get("array")
        if sl is not None:
            data = np.array(data[sl])
        else:
            data = np.array(data)
        if to_torch:
            data = torch.from_numpy(data)
            data = data.to(dtype=torch.float)
        return data


class T4C_dataset(Dataset):
    def __init__(
        self,
        image_size,
        all_files: list,
        seq_len: int = 12,
        num_shards: int = 1,
        shard: int = 0,
    ):
        """torch dataset from training data.
        Parameters
        ----------
        root_dir
            data root folder, by convention should be `data/raw`, see `data/README.md`. All `**/training/*8ch.h5` will be added to the dataset.
        file_filter: str
            filter files under `root_dir`, defaults to `"**/training/*ch8.h5`
        limit
            truncate dataset size
        transform
            transform applied to both the input and label
        """
        self.files = all_files
        self.seq_len = seq_len

        self.len = len(self.files)*MAX_TEST_SLOT_INDEX
        #self._load_dataset()

    def _load_dataset(self):
        self.file_list = list(Path(self.root_dir).rglob(self.file_filter))

        self.file_list.sort()
        self.len = len(self.file_list) * MAX_TEST_SLOT_INDEX

    def _load_h5_file(self, fn, sl: Optional[slice]):
        return load_h5_file(fn, sl=sl)

    def __len__(self):

        return self.len


    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        if idx > self.__len__():
            raise IndexError("Index out of bounds")

        file_idx = idx // MAX_TEST_SLOT_INDEX
        start_hour = idx % MAX_TEST_SLOT_INDEX
        input_data = self._load_h5_file(self.files[file_idx], sl=slice(start_hour, start_hour + self.seq_len))
        #two_hours = self.files[file_idx][start_hour:start_hour+24]

        #input_data, output_data = prepare_test(two_hours)
        #input_data, output_data = two_hours[self.in_frames], two_hours[self.out_frames]

        input_data = input_data[:,128:128+128,128:128+128, 1::2]
        #output_data = output_data[:,128:128+128, 128:128+128, 0::2]
        #input_data = input_data[:,:,:, self.ch_start:self.ch_end]
        #output_data = output_data[:,:,:,self.ch_start:self.ch_end]
        input_data = 2*((input_data) / ( 255.0)) - 1

        return input_data


def _list_video_files_recursively(data_dir):
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["gif", "avi", "mp4"]:
            results.append(full_path)
        elif bf.isdir(full_path):
            results.extend(_list_video_files_recursively(full_path))
    return results

class VideoDataset_mp4(Dataset):
    def __init__(self, resolution, video_paths, classes=None, shard=0, num_shards=1, rgb=True, seq_len=20):
        super().__init__()
        self.resolution = resolution
        self.local_videos = video_paths[shard:][::num_shards]
        self.local_classes = None if classes is None else classes[shard:][::num_shards]
        self.rgb = rgb
        self.seq_len = seq_len

    def __len__(self):
        return len(self.local_videos)

    def __getitem__(self, idx):
        path = self.local_videos[idx]
        arr_list = []
        video_container = av.open(path)
        n = video_container.streams.video[0].frames
        frames = [i for i in range(n)]
        if n > self.seq_len:
            start = np.random.randint(0, n-self.seq_len)
            frames = frames[start:start + self.seq_len]
        for id, frame_av in enumerate(video_container.decode(video=0)):
        # We are not on a new enough PIL to support the `reducing_gap`
        # argument, which uses BOX downsampling at powers of two first.
        # Thus, we do it by hand to improve downsample quality.
            if (id not in frames):
                continue
            frame = frame_av.to_image()
            while min(*frame.size) >= 2 * self.resolution:
                frame = frame.resize(
                    tuple(x // 2 for x in frame.size), resample=Image.BOX
                )
            scale = self.resolution / min(*frame.size)
            frame =frame.resize(
                tuple(round(x * scale) for x in frame.size), resample=Image.BICUBIC
            )

            if self.rgb:
                arr = np.array(frame.convert("RGB"))
            else:
                arr = np.array(frame.convert("L"))
                arr = np.expand_dims(arr, axis=2)
            crop_y = (arr.shape[0] - self.resolution) // 2
            crop_x = (arr.shape[1] - self.resolution) // 2
            arr = arr[crop_y : crop_y + self.resolution, crop_x : crop_x + self.resolution]
            arr = arr.astype(np.float32) / 127.5 - 1
            arr_list.append(arr)
        arr_seq = np.array(arr_list)
        arr_seq = np.transpose(arr_seq, [3, 0, 1, 2])
        # fill in missing frames with 0s
        if arr_seq.shape[1] < self.seq_len:
            required_dim = self.seq_len - arr_seq.shape[1]
            fill = np.zeros((3, required_dim, self.resolution, self.resolution))
            arr_seq = np.concatenate((arr_seq, fill), axis=1)
        out_dict = {}
        if self.local_classes is not None:
            out_dict["y"] = np.array(self.local_classes[idx], dtype=np.int64)

        return arr_seq, out_dict

class VideoDataset_gif(Dataset):
    def __init__(self, resolution, video_paths, classes=None, shard=0, num_shards=1, rgb=True, seq_len=20):
        super().__init__()
        self.resolution = resolution
        self.local_videos = video_paths[shard:][::num_shards]
        self.local_classes = None if classes is None else classes[shard:][::num_shards]
        self.rgb = rgb
        self.seq_len = seq_len

    def __len__(self):
        return len(self.local_videos)

    def __getitem__(self, idx):
        path = self.local_videos[idx]
        with bf.BlobFile(path, "rb") as f:
            pil_videos = Image.open(f)
            arr_list = []
            for frame in ImageSequence.Iterator(pil_videos):

            # We are not on a new enough PIL to support the `reducing_gap`
            # argument, which uses BOX downsampling at powers of two first.
            # Thus, we do it by hand to improve downsample quality.
                while min(*frame.size) >= 2 * self.resolution:
                    frame = frame.resize(
                        tuple(x // 2 for x in frame.size), resample=Image.BOX
                    )
                scale = self.resolution / min(*frame.size)
                frame =frame.resize(
                    tuple(round(x * scale) for x in frame.size), resample=Image.BICUBIC
                )

                if self.rgb:
                    arr = np.array(frame.convert("RGB"))
                else:
                    arr = np.array(frame.convert("L"))
                    arr = np.expand_dims(arr, axis=2)
                crop_y = (arr.shape[0] - self.resolution) // 2
                crop_x = (arr.shape[1] - self.resolution) // 2
                arr = arr[crop_y : crop_y + self.resolution, crop_x : crop_x + self.resolution]
                arr = arr.astype(np.float32) / 127.5 - 1
                arr_list.append(arr)
        arr_seq = np.array(arr_list)
        arr_seq = np.transpose(arr_seq, [3, 0, 1, 2])
        if arr_seq.shape[1] > self.seq_len:
            start = np.random.randint(0, arr_seq.shape[1]-self.seq_len)
            arr_seq = arr_seq[:,start:start + self.seq_len]
        out_dict = {}
        if self.local_classes is not None:
            out_dict["y"] = np.array(self.local_classes[idx], dtype=np.int64)
        return arr_seq, out_dict
