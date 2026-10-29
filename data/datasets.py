'''
Build trainining/testing datasets
'''
import hashlib
import os
import json
import pickle
import time

from torchvision import datasets, transforms
from concurrent.futures import ThreadPoolExecutor
from torchvision.datasets.folder import IMG_EXTENSIONS, ImageFolder, default_loader
import torch

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import create_transform

try:
    from timm.data import TimmDatasetTar
except ImportError:
    # for higher version of timm
    from timm.data import ImageDataset as TimmDatasetTar

class INatDataset(ImageFolder):
    def __init__(self, root, train=True, year=2018, transform=None, target_transform=None,
                 category='name', loader=default_loader):
        self.transform = transform
        self.loader = loader
        self.target_transform = target_transform
        self.year = year
        # assert category in ['kingdom','phylum','class','order','supercategory','family','genus','name']
        path_json = os.path.join(
            root, f'{"train" if train else "val"}{year}.json')
        with open(path_json) as json_file:
            data = json.load(json_file)

        with open(os.path.join(root, 'categories.json')) as json_file:
            data_catg = json.load(json_file)

        path_json_for_targeter = os.path.join(root, f"train{year}.json")

        with open(path_json_for_targeter) as json_file:
            data_for_targeter = json.load(json_file)

        targeter = {}
        indexer = 0
        for elem in data_for_targeter['annotations']:
            king = []
            king.append(data_catg[int(elem['category_id'])][category])
            if king[0] not in targeter.keys():
                targeter[king[0]] = indexer
                indexer += 1
        self.nb_classes = len(targeter)

        self.samples = []
        for elem in data['images']:
            cut = elem['file_name'].split('/')
            target_current = int(cut[2])
            path_current = os.path.join(root, cut[0], cut[2], cut[3])

            categors = data_catg[target_current]
            target_current_true = targeter[categors[category]]
            self.samples.append((path_current, target_current_true))

    # __getitem__ and __len__ inherited from ImageFolder


def _scan_class_folder(folder):
    """Image paths below ``folder`` in torchvision ImageFolder order.

    Uses ``os.scandir`` and the file extension only (no per-file stat), and
    orders the result exactly like ``sorted(os.walk(...))`` + ``sorted(files)``.
    """
    found = []
    pending = [folder]
    while pending:
        directory = pending.pop()
        with os.scandir(directory) as entries:
            for entry in entries:
                if entry.name.lower().endswith(IMG_EXTENSIONS):
                    found.append((directory, entry.name))
                elif entry.is_dir(follow_symlinks=True):
                    pending.append(entry.path)
    found.sort()
    return [os.path.join(directory, name) for directory, name in found]


def scan_image_folder(root, classes, threads=None):
    """Parallel replacement for torchvision's ImageFolder scan, with progress.

    Directory listing is latency-bound (hard disks, NTFS/FUSE mounts, network
    storage), so listing class folders concurrently is much faster than the
    sequential walk while producing the identical (path, label) list.
    """
    threads = threads or int(os.environ.get("ERNET_INDEX_THREADS", 32))
    folders = [os.path.join(root, name) for name in classes]
    samples, started = [], time.time()
    step = max(1, len(folders) // 10)
    with ThreadPoolExecutor(max_workers=threads) as pool:
        for label, paths in enumerate(pool.map(_scan_class_folder, folders)):
            samples.extend((path, label) for path in paths)
            if (label + 1) % step == 0 or label + 1 == len(folders):
                print(f"[data]   {label + 1}/{len(folders)} classes, {len(samples)} "
                      f"images, {time.time() - started:.0f}s", flush=True)
    return samples


class CachedImageFolder(ImageFolder):
    """ImageFolder whose file index is cached after the first scan.

    Listing ImageNet (1.28M files) on a hard disk can take many minutes and
    torchvision prints nothing meanwhile, so the run looks frozen.  The first
    run scans once (with progress messages) and stores the (path, label)
    index in ``cache_dir``; later runs load it in about a second.  The cache
    is invalidated automatically when class folders, or files inside them,
    are added or removed (``ERNET_DATA_CACHE`` sets the cache folder).
    """

    def __init__(self, root, transform=None, cache_dir=None,
                 wait_for_index=False, wait_timeout=3 * 3600):
        root = os.path.abspath(os.path.expanduser(root))
        if not os.path.isdir(root):
            raise FileNotFoundError(
                f"Dataset directory not found: {root}. Set DATA_PATH / --data-path "
                "to the folder that contains train/ and val/.")
        cache_dir = cache_dir or os.environ.get(
            "ERNET_DATA_CACHE", os.path.expanduser("~/.cache/ernet_datasets"))
        classes = sorted(entry.name for entry in os.scandir(root) if entry.is_dir())
        # Directory mtimes change whenever files are added or removed, so the
        # key covers both the class list and the folder contents (cheap: one
        # stat per class folder, no file listing).
        stamps = [os.stat(os.path.join(root, name)).st_mtime_ns for name in classes]
        key = "|".join([root, *classes, str(max(stamps, default=0)),
                        str(sum(stamps) % (1 << 61))])
        signature = hashlib.sha1(key.encode()).hexdigest()[:16]
        cache_file = os.path.join(cache_dir, f"index_{signature}.pkl")

        if wait_for_index and not os.path.exists(cache_file):
            # Distributed run: rank 0 scans the folder; the other ranks wait for
            # its cache file instead of scanning the same disk concurrently
            # (a collective barrier would hit the 10-minute NCCL timeout).
            print(f"[data] waiting for rank 0 to index {root} ...", flush=True)
            deadline = time.time() + wait_timeout
            while not os.path.exists(cache_file):
                if time.time() > deadline:
                    raise TimeoutError(f"No index for {root} after {wait_timeout}s")
                time.sleep(5)

        if os.path.exists(cache_file):
            started = time.time()
            with open(cache_file, "rb") as handle:
                index = pickle.load(handle)
            # Same attributes as DatasetFolder, without rescanning the disk.
            self._init_from_index(root, transform, index)
            print(f"[data] {root}: loaded cached index of {len(self.samples)} images "
                  f"({len(self.classes)} classes) in {time.time() - started:.1f}s",
                  flush=True)
            return

        print(f"[data] Indexing {root} ({len(classes)} class folders, parallel). "
              "Done once; later runs load the cached index.", flush=True)
        started = time.time()
        samples = scan_image_folder(root, classes)
        if not samples:
            raise FileNotFoundError(f"No images found under {root}")
        self._init_from_index(root, transform, dict(
            classes=classes, class_to_idx={c: i for i, c in enumerate(classes)},
            samples=samples))
        print(f"[data] {root}: {len(self.samples)} images indexed in "
              f"{time.time() - started:.0f}s", flush=True)
        try:
            os.makedirs(cache_dir, exist_ok=True)
            temporary = cache_file + f".tmp{os.getpid()}"
            with open(temporary, "wb") as handle:
                pickle.dump(dict(classes=self.classes,
                                 class_to_idx=self.class_to_idx,
                                 samples=self.samples), handle,
                            protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(temporary, cache_file)
            print(f"[data] index cached at {cache_file}", flush=True)
        except OSError as error:
            print(f"[data] could not write index cache ({error}); continuing",
                  flush=True)

    def _init_from_index(self, root, transform, index):
        self.root = root
        self.transform = transform
        self.target_transform = None
        self.transforms = None
        self.loader = default_loader
        self.extensions = None
        self.classes = index["classes"]
        self.class_to_idx = index["class_to_idx"]
        self.samples = index["samples"]
        self.targets = [target for _, target in self.samples]
        self.imgs = self.samples


def _is_non_zero_rank():
    return (torch.distributed.is_available()
            and torch.distributed.is_initialized()
            and torch.distributed.get_rank() != 0)


def build_dataset(is_train, args):
    transform = build_transform(is_train, args)

    if args.data_set == 'CIFAR':
        dataset = datasets.CIFAR100(
            args.data_path, train=is_train, transform=transform)
        nb_classes = 100
    elif args.data_set == 'IMNET':
        prefix = 'train' if is_train else 'val'
        data_dir = os.path.join(args.data_path, f'{prefix}.tar')
        if os.path.exists(data_dir):
            print(f"[data] Reading tar index of {data_dir} (can take minutes on a "
                  "hard disk)", flush=True)
            dataset = TimmDatasetTar(data_dir, transform=transform)
        else:
            root = os.path.join(args.data_path, 'train' if is_train else 'val')
            dataset = CachedImageFolder(root, transform=transform,
                                        wait_for_index=_is_non_zero_rank())
        nb_classes = 1000
    elif args.data_set == 'IMNETEE':
        root = os.path.join(args.data_path, 'train' if is_train else 'val')
        dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 10
    elif args.data_set == 'FLOWERS':
        root = os.path.join(args.data_path, 'train' if is_train else 'test')
        dataset = datasets.ImageFolder(root, transform=transform)
        if is_train:
            dataset = torch.utils.data.ConcatDataset(
                [dataset for _ in range(100)])
        nb_classes = 102
    elif args.data_set == 'INAT':
        dataset = INatDataset(args.data_path, train=is_train, year=2018,
                              category=args.inat_category, transform=transform)
        nb_classes = dataset.nb_classes
    elif args.data_set == 'INAT19':
        dataset = INatDataset(args.data_path, train=is_train, year=2019,
                              category=args.inat_category, transform=transform)
        nb_classes = dataset.nb_classes
    return dataset, nb_classes


def build_transform(is_train, args):
    resize_im = args.input_size > 32
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(
                args.input_size, padding=4)
        return transform

    t = []
    if args.finetune:
        t.append(
            transforms.Resize((args.input_size, args.input_size),
                                interpolation=3)
        )
    else:
        if resize_im:
            size = int((256 / 224) * args.input_size)
            t.append(
                # to maintain same ratio w.r.t. 224 images
                transforms.Resize(size, interpolation=3),
            )
            t.append(transforms.CenterCrop(args.input_size))
    
    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)
