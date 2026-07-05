# -*- coding: utf-8 -*-
"""Vision port of freeshap's dataset layer.

- VisionReader (!VisionReader): CIFAR-10 (or other torchvision) 다운로드/로드
- VisionDataset (!VisionDataset): freeshap의 FastListDataset과 같은 인터페이스
    - get_idx_dataset(idxs, split) → list of {'image': tensor, 'label': int}
    - label_word_list = None (vision은 prompt 개념 없음)
    - len_train(), len_val() 제공
"""
import os
import numpy as np
import torch
from torch.utils.data import Dataset, Subset, DataLoader
from torchvision import datasets, transforms

from utils import InitYAMLObject


# ImageNet stats (pretrained ResNet 정규화용)
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]


def _build_transform(image_size=224):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
    ])


class _DictWrapper(Dataset):
    """torchvision Subset을 {'image', 'label', 'idx'} dict 아이템으로 감쌈 (freeshap 규약).

    ★ 성능 개선: freeshap의 probe.kernel_regression은 val 라벨을 매 콜마다
      `torch.tensor([i['label'] for i in test_set])`로 뽑음 (probe.py:502 등 6군데).
      각 iteration은 __getitem__을 호출하고, 기본 torchvision 데이터셋은 매번
      Resize(224) + ToTensor + Normalize 전체 파이프라인을 재실행함 → vision 슬로우다운의 주범.
      → 라벨은 base_ds.targets에서 곧바로 캐시 (transform 트리거 없이),
        이미지는 __init__에서 한 번 미리 transform하여 리스트로 캐시.
        RAM: 3500 × (3×224²×4B) ≈ 2GB — 여유롭게 맞음. NLP에는 영향 없음(별도 클래스).
    """
    def __init__(self, base_ds, idxs, eager_images=True, verbose=True):
        self.base = base_ds
        self.idxs = list(idxs)
        n = len(self.idxs)

        # ★ 라벨 캐시 — base_ds.targets에서 곧바로 (transform 우회)
        if hasattr(base_ds, 'targets'):
            targets = np.asarray(base_ds.targets)
            self.labels_cache = targets[self.idxs].astype(int).tolist()
        else:
            # fallback: transform 통해서 라벨 뽑음
            self.labels_cache = [int(base_ds[o][1]) for o in self.idxs]

        # ★ 이미지 캐시 — 미리 한 번씩만 transform 실행
        if eager_images:
            if verbose:
                print(f"[_DictWrapper] preloading {n} transformed images ...", flush=True)
            self.images_cache = [None] * n
            for i, orig in enumerate(self.idxs):
                self.images_cache[i], _ = self.base[orig]
            if verbose:
                print(f"[_DictWrapper] preload done ({n} images cached).", flush=True)
        else:
            self.images_cache = None   # lazy 모드

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, i):
        lab = self.labels_cache[i]
        if self.images_cache is not None:
            img = self.images_cache[i]
        else:
            img, _ = self.base[self.idxs[i]]
        return {'image': img, 'label': lab, 'idx': int(self.idxs[i])}


class VisionReader(InitYAMLObject):
    """!VisionReader — CIFAR-10 등 torchvision 데이터셋 로더.

    yaml args:
        dataset_name: 'cifar10' (default)
        datadir:      데이터셋 저장 경로 (default: ./datasets_vision)
        image_size:   입력 크기 (default: 224, ResNet-18 pretrained용)
    """
    yaml_tag = '!VisionReader'

    def __init__(self, args):
        args = dict(args)
        self.args = args
        self.dataset_name = args.get('dataset_name', 'cifar10')
        self.datadir      = args.get('datadir', './datasets_vision')
        self.image_size   = int(args.get('image_size', 224))
        os.makedirs(self.datadir, exist_ok=True)
        self._train = None
        self._val   = None

    def _ensure_loaded(self):
        if self._train is not None: return
        tf = _build_transform(self.image_size)
        if self.dataset_name == 'cifar10':
            self._train = datasets.CIFAR10(self.datadir, train=True,  download=True, transform=tf)
            self._val   = datasets.CIFAR10(self.datadir, train=False, download=True, transform=tf)
            self._num_classes = 10
        elif self.dataset_name == 'cifar100':
            self._train = datasets.CIFAR100(self.datadir, train=True,  download=True, transform=tf)
            self._val   = datasets.CIFAR100(self.datadir, train=False, download=True, transform=tf)
            self._num_classes = 100
        else:
            raise ValueError(f"unknown vision dataset: {self.dataset_name}")

    def get_train(self):
        self._ensure_loaded(); return self._train
    def get_val(self):
        self._ensure_loaded(); return self._val
    def num_classes(self):
        self._ensure_loaded(); return self._num_classes


class VisionDataset(InitYAMLObject):
    """!VisionDataset — freeshap FastListDataset의 vision 대응.

    yaml args:
        device:     ntk 계산용 device (예: cuda:0)
        batchsize:  DataLoader 배치사이즈
    yaml refs:
        data_loader: VisionReader 참조 (*id_reader)
    yaml 필드:
        prompt: False (vision은 prompt 아님, 하위 호환)
        num_labels: 10 (자동 결정도 가능하지만 명시)
    """
    yaml_tag = '!VisionDataset'

    def __init__(self, args, data_loader, prompt=False, num_labels=None, **kw):
        args = dict(args)
        self.args = args
        self.data_loader = data_loader
        self.prompt = bool(prompt)   # 항상 False (하위 호환용)
        self.device = args.get('device', 'cuda:0')
        self.batchsize = int(args.get('batchsize', 50))
        # num_labels 결정
        self.num_labels = int(num_labels) if num_labels is not None else data_loader.num_classes()
        # vision은 prompt 없음 → label_word_list 없음 (task_ntk의 prompt 분기에서 안 씀)
        self.label_word_list = None
        self.mapping = None
        self.template = None
        self.tokenizer = None

    # ---- freeshap FastListDataset 호환 인터페이스 ----
    def _split_dataset(self, split):
        if split == 'train':
            return self.data_loader.get_train()
        elif split in ('val', 'validation', 'test'):
            return self.data_loader.get_val()
        else:
            raise ValueError(f"unknown split={split}")

    def get_idx_dataset(self, idxs, split="train"):
        """freeshap 규약: split의 idxs 서브셋을 dict-item Dataset으로 반환."""
        base = self._split_dataset(split)
        return _DictWrapper(base, idxs)

    def get_idx_dataloader(self, idxs, split="train", batch_size=None, shuffle=False):
        ds = self.get_idx_dataset(idxs, split)
        return DataLoader(ds, batch_size=batch_size or self.batchsize, shuffle=shuffle,
                          num_workers=0, collate_fn=_collate_dict)

    def len_train(self):
        return len(self.data_loader.get_train())
    def len_val(self):
        return len(self.data_loader.get_val())


def _collate_dict(batch):
    """dict 아이템 배치를 tensor 딕셔너리로 스택."""
    images = torch.stack([b['image'] for b in batch])
    labels = torch.tensor([b['label'] for b in batch], dtype=torch.long)
    idxs   = torch.tensor([b['idx']   for b in batch], dtype=torch.long)
    return {'image': images, 'label': labels, 'idx': idxs}
