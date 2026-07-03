# -*- coding: utf-8 -*-
"""NTKVisionProbe — freeshap NTKProbe의 vision 대응.

세팅: ImageNet-pretrained ResNet + fc를 Linear(feat, 1) 스칼라 head로 교체.
학습 가능(=NTK 그래디언트 계산 대상) 파라미터:
  - unfreeze_layers 옵션으로 선택:
      "fc"                     → fc만                       (~513 params, rank 낮음)
      "fc+layer4"              → fc + layer4 전체           (~8.4M params, rank full)
      "fc+layer3+layer4"       → fc + layer3, layer4        (~10.5M params)
      "all"                    → 전체                        (~11.2M params)

compute_ntk: chunked per-sample Jacobian
  - 파라미터 축(P)을 grad_chunksize 단위로 잘라 다중 패스
  - 매 패스: n_all개 샘플의 그래디언트 slice → J_chunk [n_all, chunk_size]
  - NTK += J_chunk @ J_chunk[:n_train].t()  로 누적
  - 최종 NTK shape: [1, n_train + n_test, n_train]  (freeshap single_kernel 규약)

approximate/eigen/nystrom/kernel_regression 등 하류 로직은 NTKProbe에서 상속.
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models
from easydict import EasyDict as edict
from tqdm import tqdm

from probe import Probe, NTKProbe


# unfreeze_layers 프리셋
_UNFREEZE_PRESETS = {
    "fc":                    ("fc.",),
    "fc+layer4":             ("fc.", "layer4."),
    "fc+layer3+layer4":      ("fc.", "layer3.", "layer4."),
    "fc+layer2+layer3+layer4":("fc.", "layer2.", "layer3.", "layer4."),
    "all":                   None,          # 전부 학습가능
}


class NTKVisionProbe(NTKProbe):
    """!NTKVisionProbe — chunked per-sample Jacobian NTK for vision.

    yaml args:
        device: cuda:0
        seed: 2024
        model: 'resnet-18_pretrained' | 'resnet-18_init' | ...
        unfreeze_layers: 'fc' | 'fc+layer4' (default) | 'fc+layer3+layer4' | 'all'
        grad_chunksize: 파라미터 축 chunk 크기 (default 2_000_000)
                        너무 크면 RAM 부족, 너무 작으면 다중패스로 느려짐
        image_batch: 그래디언트 계산 시 GPU 배치 사이즈 (default 16)
        single_kernel: True 권장
        debug/correction: False
    """
    yaml_tag = '!NTKVisionProbe'

    def __init__(self, args, num_labels):
        # NLP 코드 우회: NTKProbe.__init__ 대신 Probe.__init__.
        Probe.__init__(self)

        args = edict(args)
        self.args = args
        self.num_labels = int(num_labels)
        self.device = args['device']

        # ---- 모델 로드 + 스칼라 fc ----
        model_name = args['model']
        self.model = _load_resnet(model_name, seed=int(args.get('seed', 2024)))
        self.model.to(self.device)

        # ---- unfreeze 프리셋 적용 (파라미터명 startswith 매치) ----
        preset = str(args.get('unfreeze_layers', 'fc+layer4'))
        prefixes = _UNFREEZE_PRESETS.get(preset)
        if prefixes is None:
            for p in self.model.parameters():
                p.requires_grad = True
        else:
            for n, p in self.model.named_parameters():
                p.requires_grad = any(n.startswith(pref) for pref in prefixes)

        # ---- NTKProbe 상속용 state ----
        self.__name__ = 'NTKVisionProbe'
        self.freeze_layers = args.get('num_frozen_layers', 0)
        self.ntk = None
        self.train_labels = None
        self.debug = bool(args.get('debug', False))
        self.correction = bool(args.get('correction', False))
        self.single_kernel = bool(args.get('single_kernel', True))
        self.approximate_ntk = None
        self.pre_inv = None
        self.args['signgd'] = False
        self.normalize = False

        # Eigen
        self.eigen_rank = None
        self.eigen_lam = 1e-6
        self.eigen_solver = "cholesky"
        self.eigen_dtype = torch.float64
        self.eigen_decom_mode = "top"
        self.eigen_regression = None
        self.eigen_regression_dict = {}
        # Inv
        self.inv_lam = 1e-6
        # Nystrom
        self.nystrom_d = None
        self.nystrom_lam = 1e-6
        self.nystrom_solver = "cholesky"
        self.nystrom_dtype = torch.float64
        self.nystrom_landmark_seed = 1234
        self.nystrom_jitter = 1e-8
        self.nystrom_regression = None
        self.nystrom_regression_dict = {}

        p_all = sum(p.numel() for p in self.model.parameters())
        p_trn = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        trn_names = sorted({n.split('.')[0] for n, p in self.model.named_parameters() if p.requires_grad})
        print(f"[NTKVisionProbe] model={model_name}  device={self.device}")
        print(f"[NTKVisionProbe] unfreeze={preset}  trainable_modules={trn_names}")
        print(f"[NTKVisionProbe] params(total)={p_all:,}  params(trainable)={p_trn:,}")

    # ---------------------------------------------------------------
    # compute_ntk — chunked per-sample Jacobian
    # ---------------------------------------------------------------
    def compute_ntk(self, train_set, test_set):
        self.model.eval()   # BN은 running stat 사용, gradient는 여전히 흐름
        n_train = len(train_set); n_test = len(test_set); n_all = n_train + n_test
        print(f'[NTKVisionProbe] compute_ntk  n_train={n_train}  n_test={n_test}')

        named_params = [(n, p) for n, p in self.model.named_parameters() if p.requires_grad]
        P = sum(p.numel() for _, p in named_params)

        chunk_size = int(self.args.get('grad_chunksize', 2_000_000))
        n_chunks = (P + chunk_size - 1) // chunk_size
        img_batch = int(self.args.get('image_batch', 16))
        print(f'[NTKVisionProbe] trainable P={P:,}  chunk_size={chunk_size:,}  n_chunks={n_chunks}  img_batch={img_batch}')

        # 파라미터 flat offsets (chunk 슬라이싱용)
        param_offsets = [0]
        for _, p in named_params:
            param_offsets.append(param_offsets[-1] + p.numel())

        # 모든 이미지를 CPU에 미리 stack (재사용 위해; ~3500 × 3 × 224² × 4 ≈ 2 GB)
        all_images = torch.stack([train_set[i]['image'] for i in range(n_train)] +
                                 [test_set[i]['image']  for i in range(n_test)])

        ntk_acc = torch.zeros(n_all, n_train, dtype=torch.float32)

        for ci in range(n_chunks):
            c_start = ci * chunk_size
            c_end   = min(c_start + chunk_size, P)
            c_len   = c_end - c_start

            J_chunk = torch.empty(n_all, c_len, dtype=torch.float32)

            # 이 chunk에 걸치는 파라미터들 (name, local_start, local_end, dest_start_in_chunk)
            slice_plan = []
            for (name, p), off_lo, off_hi in zip(named_params, param_offsets[:-1], param_offsets[1:]):
                if off_hi <= c_start or off_lo >= c_end: continue
                ls = max(0, c_start - off_lo)
                le = min(p.numel(), c_end - off_lo)
                ds = off_lo + ls - c_start
                slice_plan.append((name, p, ls, le, ds))

            # 이미지 배치 단위로 per-sample gradient (배치 안에선 sample 개별 backward)
            for bs in tqdm(range(0, n_all, img_batch), desc=f'chunk {ci+1}/{n_chunks}'):
                be = min(bs + img_batch, n_all)
                for i in range(bs, be):
                    self.model.zero_grad(set_to_none=True)
                    x = all_images[i].unsqueeze(0).to(self.device, non_blocking=True)
                    out = self.model(x)          # [1, 1] 스칼라
                    out.sum().backward()
                    # chunk 조각만 복사
                    for name, p, ls, le, ds in slice_plan:
                        g = p.grad.reshape(-1)
                        J_chunk[i, ds:ds+(le-ls)] = g[ls:le].detach().cpu()

            ntk_acc += J_chunk @ J_chunk[:n_train].t()
            del J_chunk
            if torch.cuda.is_available(): torch.cuda.empty_cache()

        ntk_acc = ntk_acc.unsqueeze(0).contiguous()   # [1, n_all, n_train]
        print(f'[NTKVisionProbe] ntk.shape = {tuple(ntk_acc.shape)}  '
              f'mean|ntk|={ntk_acc.abs().mean():.4f}  max={ntk_acc.max():.2f}')

        self.ntk = ntk_acc
        self.train_labels = torch.tensor(
            [train_set[i]['label'] for i in range(n_train)], dtype=torch.long)
        if self.normalize:
            self.ntk = self.ntk / 10000.0
        return ntk_acc


# =============================================================================
# 헬퍼
# =============================================================================
def _load_resnet(name: str, seed: int = 2024):
    torch.manual_seed(seed)
    if name == 'resnet-18_pretrained':
        m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT); feat = 512
    elif name == 'resnet-18_init':
        m = models.resnet18(weights=None); feat = 512
    elif name == 'resnet-34_pretrained':
        m = models.resnet34(weights=models.ResNet34_Weights.DEFAULT); feat = 512
    elif name == 'resnet-34_init':
        m = models.resnet34(weights=None); feat = 512
    elif name == 'resnet-50_pretrained':
        m = models.resnet50(weights=models.ResNet50_Weights.DEFAULT); feat = 2048
    elif name == 'resnet-50_init':
        m = models.resnet50(weights=None); feat = 2048
    else:
        raise ValueError(f"unknown vision model: {name}")
    m.fc = nn.Linear(feat, 1)
    return m
