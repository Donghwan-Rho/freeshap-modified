import torch
import torch.nn as nn
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import pathlib
import threading
import time

from .ntk import load_ntk, save_ntk

try:
    from scipy.sparse.linalg import eigsh as _eigsh
except Exception:
    _eigsh = None


################################### Kernel regression with Eigen Decomposition ########################################
class EigenNTKRegression(nn.Module):
    def __init__(self, ntk_full, y_train, n_class, rank=512, lam=1e-6, solver="cholesky", dtype=torch.float32, device='cuda:0', eigen_decom_mode="top", seed=2023):
        # print(f'>>> dtype={dtype}, device={device}')
        """
        Kernel ridge regression using eigen decomposition for efficient computation.
        
        Parameters
        ----------
        ntk_full : torch.Tensor
            Full NTK matrix of shape [1, n_total, n_train] or [n_total, n_train]
        y_train : torch.Tensor
            Training labels
        n_class : int
            Number of classes
        rank : int
            Number of eigenvalues/eigenvectors to use
        lam : float
            Ridge regularization parameter
        solver : str
            'cholesky' or 'lstsq'
        dtype : torch.dtype
            Data type for computation (float64 recommended)
        device : str or torch.device
            Device for computation
        eigen_decom_mode : str
            'top' (use largest eigenvalues) or 'random' (use random eigenvalues)
        seed : int
            Random seed for 'random' mode
        """
        super(EigenNTKRegression, self).__init__()
        self.y = y_train
        self.n_class = n_class
        self.rank = rank
        self.lam = lam
        self.solver = solver
        self.dtype = dtype
        self.device = torch.device(device) if isinstance(device, str) else device
        self.eigen_decom_mode = eigen_decom_mode
        self.seed = seed
        
        # Prepare eigen features
        self.phi_tr, self.phi_te = self._precompute_eigen_features(ntk_full)
        
    def _precompute_eigen_features(self, ntk_full):
        """
        Compute eigen decomposition and feature matrices.
        
        Parameters
        ----------
        ntk_full : torch.Tensor
            Full NTK matrix [1, n_total, n_train] or [n_total, n_train]
            
        Returns
        -------
        phi_tr : torch.Tensor
            Train feature matrix [n_train, d]
        phi_te : torch.Tensor
            Test feature matrix [n_test, d]
        """
        print("[EigenNTKRegression] Computing eigen decomposition...")
        t0 = time.time()
        
        # Handle shape
        if ntk_full.ndim == 3:
            ntk0 = ntk_full[0].detach()
        else:
            ntk0 = ntk_full.detach()
            
        ntk0 = ntk0.to("cpu", dtype=torch.float64)
        n_total, n_train = ntk0.shape
        n_test = n_total - n_train
        
        K_trtr = ntk0[:n_train, :].numpy()
        K_tetr = ntk0[n_train:, :].numpy()
        K_sym = 0.5 * (K_trtr + K_trtr.T)
        
        d = min(self.rank, n_train)
        
        # Eigen decomposition
        if n_train <= 5000 or _eigsh is None:
            evals, evecs = np.linalg.eigh(K_sym)
            
            if self.eigen_decom_mode == "top":
                idx = np.argsort(evals)[::-1][:d]
                method = f"dense_eigh(top-{d})"
            elif self.eigen_decom_mode == "random":
                # Use separate RandomState to avoid affecting global random state
                rng = np.random.RandomState(self.seed)
                idx = rng.choice(len(evals), size=d, replace=False)
                method = f"dense_eigh(random-{d})"
            else:
                raise ValueError(f"Unknown eigen_decom_mode: {self.eigen_decom_mode}")
            
            lam = evals[idx]
            U = evecs[:, idx]
        else:
            if self.eigen_decom_mode == "random":
                # For random mode with large matrix, compute all then select random
                evals, evecs = np.linalg.eigh(K_sym)
                rng = np.random.RandomState(self.seed)
                idx = rng.choice(len(evals), size=d, replace=False)
                lam = evals[idx]
                U = evecs[:, idx]
                method = f"dense_eigh(random-{d})"
            else:
                # top mode with sparse solver
                lam, U = _eigsh(K_sym, k=d, which='LA', tol=1e-4)
                idx = np.argsort(lam)[::-1]
                lam = lam[idx]
                U = U[:, idx]
                method = f"eigsh(top-{d})"
        
        lam = np.clip(lam, 1e-8, None)
        
        # Compute feature matrices
        Phi_tr = U * np.sqrt(lam)[None, :]
        Phi_te = K_tetr @ (U / np.sqrt(lam)[None, :])
        
        print(f"[EigenNTKRegression] Done via {method} (n_train={n_train}, d={d}, d/n_train(%)={d/n_train*100:.2f}%) in {time.time()-t0:.2f}s")
        
        # Move to device
        phi_tr = torch.from_numpy(Phi_tr).to(device=self.device, dtype=self.dtype)
        phi_te = torch.from_numpy(Phi_te).to(device=self.device, dtype=self.dtype)
        
        return phi_tr, phi_te
    
    def _robust_cholesky(self, A, base_reg=0.0, max_tries=6, growth=10.0):
        """Robust Cholesky decomposition with increasing regularization."""
        device, dtype = A.device, A.dtype
        n = A.size(0)
        I = torch.eye(n, device=device, dtype=dtype)
        reg = float(base_reg)
        
        for _ in range(max_tries):
            L, info = torch.linalg.cholesky_ex(A + reg * I)
            if info == 0:
                return L, reg
            reg = max(1e-12, reg * growth) if reg > 0 else 1e-8
        
        # Final attempt
        reg = max(reg, 1e-2)
        L, info = torch.linalg.cholesky_ex(A + reg * I)
        if info != 0:
            raise RuntimeError("robust_cholesky failed")
        return L, reg
    
    def _ridge_solve(self, PhiS, YS, effective_lam=None):
        """
        Solve ridge regression: min_W ||PhiS @ W - YS||^2 + lam * ||W||^2
        
        Parameters
        ----------
        PhiS : torch.Tensor [m, d]
        YS : torch.Tensor [m, C]
        effective_lam : float, optional
            Override lambda value for adaptive regularization
        
        Returns
        -------
        W : torch.Tensor [d, C]
        """
        lam_to_use = effective_lam if effective_lam is not None else self.lam
        
        if self.solver == "lstsq":
            device, dtype = PhiS.device, PhiS.dtype
            d = PhiS.size(1)
            if lam_to_use > 0:
                sq = torch.as_tensor(lam_to_use, device=device, dtype=dtype).sqrt()
                A = torch.cat([PhiS, sq * torch.eye(d, device=device, dtype=dtype)], dim=0)
                Z = torch.cat([YS, torch.zeros(d, YS.size(1), device=device, dtype=dtype)], dim=0)
            else:
                A, Z = PhiS, YS
            return torch.linalg.lstsq(A, Z).solution
        
        # Cholesky solver
        d = PhiS.size(1)
        B = PhiS.T @ PhiS
        RHS = PhiS.T @ YS
        
        try:
            L, used_reg = self._robust_cholesky(B, base_reg=lam_to_use)
            return torch.cholesky_solve(RHS, L)
        except Exception as e:
            print(f"[WARN] Cholesky failed ({e}); fallback to lstsq")
            device, dtype = PhiS.device, PhiS.dtype
            if lam_to_use > 0:
                sq = torch.as_tensor(lam_to_use, device=device, dtype=dtype).sqrt()
                A = torch.cat([PhiS, sq * torch.eye(d, device=device, dtype=dtype)], dim=0)
                Z = torch.cat([YS, torch.zeros(d, YS.size(1), device=device, dtype=dtype)], dim=0)
            else:
                A, Z = PhiS, YS
            return torch.linalg.lstsq(A, Z).solution
    
    def forward(self, train_indices):
        """
        Perform kernel ridge regression on subset of training data.
        
        Parameters
        ----------
        train_indices : numpy.ndarray or list
            Indices of training samples to use
            
        Returns
        -------
        test_preds : torch.Tensor [n_test, n_class]
            Predictions on test set
        """
        idx_t = torch.as_tensor(train_indices, device=self.device, dtype=torch.long)
        
        PhiS = self.phi_tr.index_select(0, idx_t)
        yS = self.y[train_indices].to(device=self.device, dtype=torch.long)
        
        # One-hot encoding
        YS = torch.nn.functional.one_hot(yS, num_classes=self.n_class).to(dtype=self.dtype)
        
        # Use constant lambda (no adaptive regularization)
        effective_lam = self.lam
        
        # Solve ridge regression
        W = self._ridge_solve(PhiS, YS, effective_lam=effective_lam)
        
        # Predict on test set
        test_preds = (self.phi_te @ W).to('cpu')
        
        return test_preds


################################### Kernel regression with Dynamic Programming INVerse ################################
class shapleyNTKRegression(nn.Module):
    def __init__(self, k_train, y, n_class, pre_inv=None, reg=1e-6):
        # print(f'[DEBUG] vinfo/entks/ntk_regression.py """shapleyNTKRegression"""')
        # print(f'k_train: {k_train.shape}')
        # print(f'y: {y.shape}')
        # print(f'n_class: {n_class}')
        # print(f'pre_inv: {pre_inv}')
        """
        Parameters
        ----------
        k_train
        y
        n_class
        pre_inv
        reg: regularization parameter (lambda)
        batch_size: checkpoint when less than this compute the exact inverse
                                    greater than this compute the exact inverse only at this position
        """
        super(shapleyNTKRegression, self).__init__()
        # print(f'[DEBUG] inv mode, reg: {reg}')
        self.y = y.double()
        self.n_class = n_class
        n = k_train.size(1)
        identity = torch.eye(n, device=k_train.device).unsqueeze(0)
        self.k_train = k_train + identity * reg
        self.pre_inv = pre_inv
        self.single_kernel = False
        if k_train.size(0) == 1:
            self.single_kernel = True
        self.k_train = self.k_train.double()

    def forward(self, k_test, return_inv=False):
        preds = []
        for i in range(self.n_class):
            # print(f'[DEBUG] i: {i}')
            yi = torch.clone(self.y)
            yi[self.y==i] = 1
            yi[self.y!=i] = 0
            # compute inverse or use cached inverse
            if self.pre_inv is not None:
                # https://en.wikipedia.org/wiki/Block_matrix
                n = self.k_train.size(1)
                # Extract sub matrices
                B = self.k_train[0][:n - 1, n - 1:]
                C = self.k_train[0][n - 1:, :n - 1]
                D = self.k_train[0][n - 1:, n - 1:]

                A_inv = self.pre_inv
                D_minus_CA_invB = D - torch.mm(C, torch.mm(A_inv, B))

                P = torch.cat((
                    torch.cat((A_inv + torch.mm(torch.mm(A_inv, B),
                                                torch.mm(torch.inverse(D_minus_CA_invB), torch.mm(C, A_inv))),
                               -torch.mm(A_inv, torch.mm(B, torch.inverse(D_minus_CA_invB)))), dim=1),
                    torch.cat((-torch.mm(torch.mm(torch.inverse(D_minus_CA_invB), C), A_inv),
                               torch.inverse(D_minus_CA_invB)), dim=1)
                ), dim=0)
                beta_i = P @ yi
            else: # 여기
                # solve linear systems
                if self.single_kernel:
                    beta_i = torch.linalg.solve(self.k_train[0], yi)
                else:
                    beta_i = torch.linalg.solve(self.k_train[i], yi)

            if self.single_kernel:
                pred_i = k_test[0].double() @ beta_i
            else:
                pred_i = k_test[i].double() @ beta_i
            preds.append(pred_i)
        y_pred = torch.stack(preds, dim=1)
        # print(f'y_pred: {y_pred.shape}')
        if return_inv:
            if self.pre_inv is not None:
                return y_pred, P
            else:
                if self.single_kernel:
                    return y_pred, torch.inverse(self.k_train[0])
                else:
                    return y_pred, torch.inverse(self.k_train[i])
        else:
            return y_pred


################################### Fast Kernel regression approximation with block-inverse ############################
class fastNTKRegression(nn.Module):
    def __init__(self, k_train, y, n_class, inv=None, batch_size=None):
        super(fastNTKRegression, self).__init__()
        self.y = y.double()
        self.n_class = n_class
        if inv is None:
            n = k_train.size(1)
            identity = torch.eye(n, device=k_train.device).unsqueeze(0)
            reg = 1e-6
            self.k_train = k_train + identity * reg
            self.inv = None
        else:
            self.inv = inv
        self.single_kernel = False
        if k_train.size(0) == 1:
            self.single_kernel = True
        self.k_train = self.k_train.double()
        self.batch_size = batch_size

    def forward(self, k_test):
        preds = []
        for i in range(self.n_class):
            yi = torch.clone(self.y)
            yi[self.y==i] = 1
            yi[self.y!=i] = 0
            # compute inverse or use cached inverse
            if self.inv is not None:
                beta_i = self.inv @ yi
            else:
                # solve linear systems
                if self.single_kernel:
                    if self.batch_size is not None:
                        beta_i = []
                        for j in range(0, self.k_train.size(1), self.batch_size):
                            H = self.k_train[0, j:j+self.batch_size, j:j+self.batch_size]
                            beta_j = torch.linalg.solve(H, yi[j:j + self.batch_size])
                            beta_i.append(beta_j)
                        beta_i = torch.cat(beta_i, dim=0)
                else:
                    # improvement notes: does not support multi kernel block diagonal but can be easily solved
                    beta_i = torch.linalg.solve(self.k_train[i], yi)
            if self.single_kernel:
                pred_i = k_test[0].double() @ beta_i
            else:
                pred_i = k_test[i].double() @ beta_i
            preds.append(pred_i)
        y_pred = torch.stack(preds, dim=1)
        return y_pred


################################### Standard Kernel regression ##############################################
class NTKRegression(nn.Module):
    def __init__(self, k_train, y, n_class, inv=None):
        super(NTKRegression, self).__init__()
        self.y = y.double()
        self.n_class = n_class
        if inv is None:
            n = k_train.size(1)
            identity = torch.eye(n, device=k_train.device).unsqueeze(0)
            reg = 1e-6
            self.k_train = k_train + identity * reg
            self.inv = None
        else:
            self.inv = inv
        self.single_kernel = False
        if k_train.size(0) == 1:
            self.single_kernel = True
        self.k_train = self.k_train.double()

    def forward(self, k_test):
        preds = []
        for i in range(self.n_class):
            yi = torch.clone(self.y) 
            yi[self.y==i] = 1
            yi[self.y!=i] = 0
            # compute inverse or use cached inverse
            if self.inv is not None:
                beta_i = self.inv @ yi 
            else:
                # solve linear systems
                if self.single_kernel:
                    try:
                        beta_i = torch.linalg.solve(self.k_train[0], yi)
                    except:
                        print("Singular matrix")
                        beta_i = torch.linalg.lstsq(self.k_train[0], yi.unsqueeze(1)).solution.squeeze()

                else:
                    try:
                        beta_i = torch.linalg.solve(self.k_train[i], yi)
                    except:
                        print("Singular matrix")
                        beta_i = torch.linalg.lstsq(self.k_train[i], yi.unsqueeze(1)).solution.squeeze()
            if self.single_kernel:
                pred_i = k_test[0].double() @ beta_i
            else:
                pred_i = k_test[i].double() @ beta_i
            preds.append(pred_i)
        y_pred = torch.stack(preds, dim=1)
        return y_pred


############################# Kernel regression with correction when multiple class ####################################
class NTKRegression_correction_multiclass(nn.Module):
    def __init__(self, k_train, y, n_class, train_logits, test_logits, inv=None):
        super(NTKRegression_correction_multiclass, self).__init__()
        self.y = y.double()
        self.n_class = n_class
        if inv is None:
            n = k_train.size(1)
            identity = torch.eye(n, device=k_train.device).unsqueeze(0)
            reg = 1e-6
            self.k_train = k_train + identity * reg
            self.inv = None
        else:
            self.inv = inv
        self.train_logits = train_logits
        self.test_logits = test_logits
        self.single_kernel = False
        self.k_train = self.k_train.double()
        if k_train.size(0) == 1:
            self.single_kernel = True

    def forward(self, k_test):
        preds = []
        for i in range(self.n_class):
            yi = torch.clone(self.y)
            yi[self.y==i] = 1
            yi[self.y!=i] = 0
            # add correction
            yi = yi - self.train_logits[:, i]
            if self.single_kernel:
                i = 0
            # compute inverse or use cached inverse
            if self.inv is not None:
                beta_i = self.inv @ yi
            else:
                try:
                    beta_i = torch.linalg.solve(self.k_train[i], yi)
                except:
                    print("Singular matrix")
                    beta_i = torch.linalg.lstsq(self.k_train[i], yi.unsqueeze(1)).solution.squeeze()

            pred_i = k_test[i].double() @ beta_i
            preds.append(pred_i)

        y_pred = torch.stack(preds, dim=1)
        try:
            y_pred = y_pred + self.test_logits
        except:
            pass
        return y_pred