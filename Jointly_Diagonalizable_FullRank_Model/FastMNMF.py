#! /usr/bin/env python3
# coding: utf-8

import numpy as np
import chainer
import sys, os
import soundfile as sf
import time
import pickle as pic

from configure_FastModel import *
from FastFCA import FastFCA

from scipy.signal import stft, istft, get_window
try:
    from chainer import cuda
    FLAG_GPU_Available = True
except:
    print("---Warning--- You cannot use GPU acceleration because chainer or cupy is not installed")


class FastMNMF(FastFCA):

    def __init__(self, NUM_source=2, NUM_basis=8, xp=np, MODE_initialize_covarianceMatrix="unit"):
        """ initialize FastMNMF

        Parameters:
        -----------
            NUM_source: int
                the number of sources
            NUM_iteration: int
                the number of iteration to update all variables
            NUM_basis: int
                the number of bases of each source
            MODE_initialize_covarianceMatrix: str
                how to initialize covariance matrix {unit, obs}
        """
        super(FastMNMF, self).__init__(NUM_source=NUM_source, xp=xp, MODE_initialize_covarianceMatrix=MODE_initialize_covarianceMatrix)
        self.NUM_basis = NUM_basis
        self.method_name = "FastMNMF"


    def set_parameter(self, NUM_source=None, NUM_basis=None, MODE_initialize_covarianceMatrix=None):
        """ set parameters

        Parameters:
        -----------
            NUM_source: int
            NUM_iteration: int
            NUM_basis: int
            MODE_initialize_covarianceMatrix: str
                how to initialize covariance matrix {unit, obs}
        """
        super(FastMNMF, self).set_parameter(NUM_source=NUM_source, MODE_initialize_covarianceMatrix=MODE_initialize_covarianceMatrix)
        if NUM_basis != None:
            self.NUM_basis = NUM_basis


    def initialize_PSD(self):
        power_observation_FT = (self.xp.abs(self.X_FTM).astype(self.xp.float32) ** 2).mean(axis=2)
        shape = 2
        self.W_NFK = self.xp.random.dirichlet(np.ones(self.NUM_freq)*shape, size=[self.NUM_source, self.NUM_basis]).transpose(0, 2, 1)
        self.H_NKT = self.xp.random.gamma(shape, (power_observation_FT.mean() * self.NUM_freq * self.NUM_mic / (self.NUM_source * self.NUM_basis)) / shape, size=[self.NUM_source, self.NUM_basis, self.NUM_time])
        self.H_NKT[self.H_NKT < EPS] = EPS
        self.lambda_NFT = self.W_NFK @ self.H_NKT


    def make_fileName_suffix(self):
        self.fileName_suffix = "S={}-it={}-L={}-init={}".format(self.NUM_source, self.NUM_iteration, self.NUM_basis, self.MODE_initialize_covarianceMatrix)

        if hasattr(self, "file_id"):
            self.fileName_suffix += "-ID={}".format(self.file_id)
        else:
            print("====================\n\nWarning: Please set self.file_id\n\n====================")

        print("fileName_suffix:", self.fileName_suffix)


    def update(self):
        self.update_WH()
        self.update_CovarianceDiagElement()
        self.update_Diagonalizer()
        self.normalize()


    def update_WH(self):
        tmp1_NFT = (self.covarianceDiag_NFM[:, :, None] * (self.Qx_power_FTM / (self.Y_FTM ** 2))[None]).sum(axis=3)
        tmp2_NFT = (self.covarianceDiag_NFM[:, :, None] / self.Y_FTM[None]).sum(axis=3)
        a_W = (self.H_NKT[:, None] * tmp1_NFT[:, :, None]).sum(axis=3)  # N F K T M
        b_W = (self.H_NKT[:, None] * tmp2_NFT[:, :, None]).sum(axis=3)
        a_H = (self.W_NFK[..., None] * tmp1_NFT[:, :, None] ).sum(axis=1) # N F K T M
        b_H = (self.W_NFK[..., None] * tmp2_NFT[:, :, None]).sum(axis=1) # N F K T M
        self.W_NFK = self.W_NFK * self.xp.sqrt(a_W / b_W)
        self.H_NKT = self.H_NKT * self.xp.sqrt(a_H / b_H)

        self.lambda_NFT = self.W_NFK @ self.H_NKT + EPS
        self.Y_FTM = (self.lambda_NFT[..., None] * self.covarianceDiag_NFM[:, :, None]).sum(axis=0)


    def normalize(self):
        phi_F = self.xp.sum(self.diagonalizer_FMM * self.diagonalizer_FMM.conj(), axis=(1, 2)).real / self.NUM_mic
        self.diagonalizer_FMM = self.diagonalizer_FMM / self.xp.sqrt(phi_F)[:, None, None]
        self.covarianceDiag_NFM = self.covarianceDiag_NFM / phi_F[None, :, None]

        mu_NF = (self.covarianceDiag_NFM).sum(axis=2).real
        self.covarianceDiag_NFM = self.covarianceDiag_NFM / mu_NF[:, :, None]
        self.W_NFK = self.W_NFK * mu_NF[:, :, None]

        nu_NK = self.W_NFK.sum(axis=1)
        self.W_NFK = self.W_NFK / nu_NK[:, None]
        self.H_NKT = self.H_NKT * nu_NK[:, :, None]
        self.lambda_NFT = self.W_NFK @ self.H_NKT + EPS

        self.reset_variable()


    def save_parameter(self, fileName):
        param_list = [self.lambda_NFT, self.covarianceDiag_NFM, self.diagonalizer_FMM, self.W_NFK, self.H_NKT]
        if self.xp != np:
            param_list = [self.convert_to_NumpyArray(param) for param in param_list]

        pic.dump(param_list, open(fileName, "wb"))


    def load_parameter(self, fileName):
        param_list = pic.load(open(fileName, "rb"))
        if self.xp != np:
            param_list = [cuda.to_gpu(param) for param in param_list]

        self.lambda_NFT, self.covarianceDiag_NFM, self.diagonalizer_FMM, self.W_NFK, self.H_NKT = param_list



if __name__ == "__main__":
    import argparse
    import pickle as pic
    import sys, os

    parser = argparse.ArgumentParser()
    parser.add_argument(    'input_fileName', type= str, help='filename of the multichannel observed signals')
    parser.add_argument(         '--file_id', type= str, default="None", help='file id')
    parser.add_argument(             '--gpu', type= int, default=    0, help='GPU ID')
    parser.add_argument(           '--n_fft', type= int, default= 1024, help='number of frequencies')
    parser.add_argument(      '--NUM_source', type= int, default=    2, help='number of noise')
    parser.add_argument(   '--NUM_iteration', type= int, default=  100, help='number of iteration')
    parser.add_argument(       '--NUM_basis', type= int, default=    8, help='number of basis')
    parser.add_argument( '--MODE_initialize_covarianceMatrix', type=  str, default="obs", help='unit, obs')
    args = parser.parse_args()

    if args.gpu < 0:
        import numpy as xp
    else:
        import cupy as xp
        print("Use GPU " + str(args.gpu))
        cuda.get_device_from_id(args.gpu).use()

    sig, fs = sf.read(args.input_fileName, always_2d=True)
    spec_FNM = np.transpose(
        stft(sig.T, window='hann', nperseg=args.n_fft, noverlap=3*args.n_fft//4)[-1],
        (1, 2, 0))
    stft_scale = get_window('hann', args.n_fft).sum()
    spec_FNM *= stft_scale

    separater = FastMNMF(NUM_source=args.NUM_source, NUM_basis=args.NUM_basis, xp=xp, MODE_initialize_covarianceMatrix=args.MODE_initialize_covarianceMatrix)
    separater.load_spectrogram(spec_FNM)
    separater.file_id = args.file_id
    separater.solve(NUM_iteration=args.NUM_iteration, save_likelihood=False, save_parameter=False, save_wav=False, save_path="./", interval_save_parameter=25)
