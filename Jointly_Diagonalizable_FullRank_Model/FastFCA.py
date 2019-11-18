#! /usr/bin/env python3
# coding: utf-8

import sys
import pickle as pic

import argparse
import numpy as np
import soundfile as sf

from scipy.signal import stft, istft, get_window
from tqdm import tqdm

from configure_FastModel import *


sys.path.append("../CupyLibrary")
try:
    from cupy_matrix_inverse import inv_gpu_batch
    FLAG_CupyInverse_Enabled = True
except:
    print("---Warning--- You cannot use cupy inverse calculation")
    FLAG_CupyInverse_Enabled = False


class FastFCA():

    def __init__(
            self, NUM_source=2, xp=np,
            MODE_initialize_covarianceMatrix="unit"):
        """ initialize FastFCA

        Parameters:
        -----------
            NUM_source: int
                the number of sources
            MODE_initialize_covarianceMatrix: str
                how to initialize covariance matrix {unit, obs}
        """
        self.NUM_source = NUM_source
        self.MODE_initialize_covarianceMatrix =\
            MODE_initialize_covarianceMatrix
        self.xp = xp
        self.calculateInverseMatrix =\
            self.return_InverseMatrixCalculationMethod()
        self.method_name = "FastFCA"

        self.NUM_freq, self.NUM_time, self.NUM_mic = None, None, None
        self.X_FTM, self.XX_FTMM = None, None

        self.lambda_NFT = None
        self.diagonalizer_FMM = None
        self.covarianceDiag_NFM = None

        self.Qx_power_FTM = None
        self.Y_FTM = None

        self.file_id = None
        self.fileName_suffix = None
        self.NUM_iteration = None
        self.separated_spec = None


    def convert_to_NumpyArray(self, data):
        if self.xp != np:
            data = self.xp.asnumpy(data)
        return data

    def return_InverseMatrixCalculationMethod(self):
        if self.xp == np:
            inv_func = np.linalg.inv
        elif FLAG_CupyInverse_Enabled:
            inv_func = inv_gpu_batch
        else:
            inv_func = lambda x: self.xp.asarray(np.linalg.inv(
                self.convert_to_NumpyArray(x)))
        return inv_func

    def set_parameter(
            self, NUM_source=None, MODE_initialize_covarianceMatrix=None):
        """ set parameters

        Parameters:
        -----------
            NUM_source: int
                the number of sources
            MODE_initialize_covarianceMatrix: str
                how to initialize covariance matrix {unit, obs}
        """
        if NUM_source is not None:
            self.NUM_source = NUM_source
        if MODE_initialize_covarianceMatrix is not None:
            self.MODE_initialize_covarianceMatrix =\
                MODE_initialize_covarianceMatrix


    def load_spectrogram(self, X_FTM):
        """ load complex spectrogram

        Parameters:
        -----------
            X_FTM: self.xp.array [ F * T * M ]
                power spectrogram of observed signals
        """
        self.NUM_freq, self.NUM_time, self.NUM_mic = X_FTM.shape
        self.X_FTM = self.xp.asarray(X_FTM, dtype=C_FP_TYPE)
        self.XX_FTMM =\
            self.X_FTM[:, :, :, None] @ self.X_FTM[:, :, None, :].conj()


    def initialize_PSD(self):
        self.lambda_NFT = self.xp.random.random(
            [self.NUM_source, self.NUM_freq, self.NUM_time]).astype(F_FP_TYPE)
        self.lambda_NFT[0] = self.xp.abs(self.X_FTM.mean(axis=2)) ** 2


    def initialize_covarianceMatrix(self):
        if "unit" in self.MODE_initialize_covarianceMatrix:
            self.diagonalizer_FMM = self.xp.tile(
                self.xp.eye(self.NUM_mic), [self.NUM_freq, 1, 1]).astype(
                    C_FP_TYPE)
            self.covarianceDiag_NFM = self.xp.ones(
                [self.NUM_source, self.NUM_freq, self.NUM_mic],
                dtype=F_FP_TYPE) / self.NUM_mic
        elif "obs" in self.MODE_initialize_covarianceMatrix:
            mixture_covarianceMatrix_FMM =\
                self.XX_FTMM.sum(axis=1) / (self.xp.trace(
                    self.XX_FTMM, axis1=2, axis2=3).sum(axis=1))[:, None, None]
            eig_val, eig_vec = np.linalg.eigh(
                self.convert_to_NumpyArray(mixture_covarianceMatrix_FMM))
            self.diagonalizer_FMM = self.xp.asarray(
                eig_vec).transpose(0, 2, 1).conj()
            self.covarianceDiag_NFM = self.xp.ones(
                [self.NUM_source, self.NUM_freq, self.NUM_mic],
                dtype=F_FP_TYPE) / self.NUM_mic
            self.covarianceDiag_NFM[0] = self.xp.asarray(eig_val)
        else:
            print("Specify how to initialize covariance matrix {unit, obs}!")
            raise ValueError

        self.normalize()


    def reset_variable(self):
        if self.xp == np:
            self.Qx_power_FTM = self.xp.abs(
                (self.diagonalizer_FMM[:, None] @ self.X_FTM[:, :, :, None])
                [:, :, :, 0]) ** 2
        else:
            self.Qx_power_FTM = self.xp.abs(
                (self.diagonalizer_FMM[:, None] * self.X_FTM[:, :, None])
                .sum(axis=3)) ** 2
        self.Y_FTM = (
            self.lambda_NFT[..., None] *
            self.covarianceDiag_NFM[:, :, None]).sum(axis=0)


    def make_fileName_suffix(self):
        self.fileName_suffix = "S={}-it={}-init={}".format(
            self.NUM_source, self.NUM_iteration,
            self.MODE_initialize_covarianceMatrix)

        if hasattr(self, "file_id"):
            self.fileName_suffix += "-ID={}".format(self.file_id)
        else:
            print("========\n\nWarning: Please set self.file_id\n\n========")

        print("parameter:", self.fileName_suffix)
        return self.fileName_suffix


    def solve(
            self, NUM_iteration=100, mic_index=MIC_INDEX,
            save_likelihood=False, save_parameter=False, save_wav=False,
            save_path="./", interval_save_parameter=30):
        """
        Parameters:
            save_likelihood: boolean
                save likelihood and lower bound or not
            save_parameter: boolean
                save parameter or not
            save_wav: boolean
                save intermediate separated signal or not
            save_path: str
                directory for saving data
            interval_save_parameter: int
                interval of saving parameter
        """
        self.NUM_iteration = NUM_iteration

        self.initialize_PSD()
        self.initialize_covarianceMatrix()
        self.make_fileName_suffix()

        log_likelihood_array = []
        for it in tqdm(range(self.NUM_iteration)):
            self.update()

            if save_parameter and ((it+1) % interval_save_parameter == 0) and\
                    ((it+1) != self.NUM_iteration):
                self.save_parameter(
                    save_path+"{}-parameter-{}-{}.pic".format(
                        self.method_name, self.fileName_suffix, it + 1))

            if save_wav and ((it+1) % interval_save_parameter == 0) and\
                    ((it+1) != self.NUM_iteration):
                self.separate_FastWienerFilter(mic_index=MIC_INDEX)
                self.save_separated_signal(
                    save_path+"{}-sep-Wiener-{}-{}.wav".format(
                        self.method_name, self.fileName_suffix, it + 1))

            if save_likelihood and ((it+1) % interval_save_parameter == 0) and\
                    ((it+1) != self.NUM_iteration):
                log_likelihood_array.append(self.calculate_log_likelihood())

        if save_parameter:
            self.save_parameter(
                save_path+"{}-parameter-{}.pic".format(
                    self.method_name, self.fileName_suffix))

        if save_likelihood:
            log_likelihood_array.append(self.calculate_log_likelihood())
            pic.dump(
                log_likelihood_array, open(
                    save_path + "{}-likelihood-interval={}-{}.pic".format(
                        self.method_name, interval_save_parameter,
                        self.fileName_suffix), "wb"))

        self.separate_FastWienerFilter(mic_index=mic_index)
        self.save_separated_signal(
            save_path+"{}-sep-Wiener-{}.wav".format(
                self.method_name, self.fileName_suffix))


    def update(self):
        self.update_lambda()
        self.update_CovarianceDiagElement()
        self.update_Diagonalizer()
        self.normalize()


    def update_Diagonalizer(self):
        # Eq. (18): V_fm = 1/T * sum_t X_ft / \tilde{y}_ftm
        V_FMMM = (
            self.XX_FTMM[:, :, None] / self.Y_FTM[:, :, :, None, None]).mean(
                axis=1)
        for m in range(self.NUM_mic):
            # Eq. (19): q_fm <- (Q_f V_fm)^{-1} e_m
            tmp_FM = self.calculateInverseMatrix(
                self.diagonalizer_FMM @ V_FMMM[:, m])[:, :, m]
            # Eq. (20): q_fm <- q_fm / sqrt(q_fm^H V_fm q_fm)
            tmp_FM /= self.xp.sqrt(self.xp.sum(
                tmp_FM.conj()[:, :, None] * V_FMMM[:, m] * tmp_FM[:, None, :],
                axis=(-2, -1)))[:, None]
            self.diagonalizer_FMM[:, m] = tmp_FM.conj()


    def update_CovarianceDiagElement(self):
        a_1 = (self.lambda_NFT[..., None] * (
            self.Qx_power_FTM / (self.Y_FTM ** 2))[None]).sum(axis=2) # N F T M
        b_1 = (self.lambda_NFT[..., None] / self.Y_FTM[None]).sum(axis=2)
        self.covarianceDiag_NFM = self.covarianceDiag_NFM * self.xp.sqrt(
            a_1 / b_1)
        self.covarianceDiag_NFM += EPS
        self.Y_FTM = (
            self.lambda_NFT[..., None] *
            self.covarianceDiag_NFM[:, :, None]).sum(axis=0)


    def update_lambda(self):
        a = (self.covarianceDiag_NFM[:, :, None] * (
            self.Qx_power_FTM / (self.Y_FTM ** 2))[None]).sum(axis=3) # N F T
        b = (
            self.covarianceDiag_NFM[:, :, None] / self.Y_FTM[None]).sum(axis=3)
        self.lambda_NFT = self.lambda_NFT * self.xp.sqrt(a / b) + EPS
        self.Y_FTM = (
            self.lambda_NFT[..., None] *
            self.covarianceDiag_NFM[:, :, None]).sum(axis=0)


    def normalize(self):
        phi_F = self.xp.sum(
            self.diagonalizer_FMM * self.diagonalizer_FMM.conj(),
            axis=(1, 2)).real / self.NUM_mic
        self.diagonalizer_FMM /= self.xp.sqrt(phi_F)[:, None, None]
        self.covarianceDiag_NFM /= phi_F[None, :, None]

        mu_NF = self.xp.sum(self.covarianceDiag_NFM, axis=2).real
        self.covarianceDiag_NFM /= mu_NF[:, :, None]
        self.lambda_NFT *= mu_NF[:, :, None]
        self.lambda_NFT += EPS

        self.reset_variable()


    def calculate_log_likelihood(self):
        return (
            -(self.Qx_power_FTM / self.Y_FTM).sum() + self.NUM_time * np.log(np.linalg.det(self.convert_to_NumpyArray(self.diagonalizer_FMM @ self.diagonalizer_FMM.conj().transpose(0, 2, 1)))).sum() - self.xp.log(self.Y_FTM).sum()).real - self.NUM_mic * self.NUM_freq * self.NUM_time * np.log(np.pi)


    def calculate_covarianceMatrix(self):
        covarianceMatrix_NFMM = self.xp.zeros(
            [self.NUM_source, self.NUM_freq, self.NUM_mic, self.NUM_mic],
            dtype=C_FP_TYPE)
        diagonalizer_inv_FMM = self.calculateInverseMatrix(
            self.diagonalizer_FMM)
        for n in range(self.NUM_source):
            for f in range(self.NUM_freq):
                covarianceMatrix_NFMM[n, f] =\
                    diagonalizer_inv_FMM[f] @ np.diag(
                        self.covarianceDiag_NFM[n, f]) @\
                    diagonalizer_inv_FMM[f].conj().T
        return covarianceMatrix_NFMM


    def separate_FastWienerFilter(self, source_index=None, mic_index=MIC_INDEX):
        Qx_FTM = xp.sum(
            self.diagonalizer_FMM[:, None] * self.X_FTM[:, :, None], axis=3)
        if source_index is not None:
            diagonalizer_inv_FMM = self.calculateInverseMatrix(self.diagonalizer_FMM)
            self.separated_spec = self.convert_to_NumpyArray((diagonalizer_inv_FMM[:, None] @ (Qx_FTM * ((self.lambda_NFT[source_index, :, :, None] * self.covarianceDiag_NFM[source_index, :, None]) / (self.lambda_NFT[..., None]* self.covarianceDiag_NFM[:, :, None]).sum(axis=0)))[..., None])[:, :, mic_index, 0])
        else:
            for n in range(self.NUM_source):
                diagonalizer_inv_FMM = self.calculateInverseMatrix(self.diagonalizer_FMM)
                tmp = self.convert_to_NumpyArray((diagonalizer_inv_FMM[:, None] @ (Qx_FTM * ((self.lambda_NFT[n, :, :, None] * self.covarianceDiag_NFM[n, :, None]) / (self.lambda_NFT[..., None]* self.covarianceDiag_NFM[:, :, None]).sum(axis=0)))[..., None])[:, :, mic_index, 0])
                if n == 0:
                    self.separated_spec = np.zeros([self.NUM_source, tmp.shape[0], tmp.shape[1]], dtype=np.complex)
                self.separated_spec[n] = tmp


    def save_separated_signal(self, save_fileName="sample.wav"):
        sep_src_stft_MFN = self.convert_to_NumpyArray(self.separated_spec)
        # infer STFT length and scale (= get_window('hann', args.n_fft).sum())
        stft_winlen = (sep_src_stft_MFN.shape[1] - 1) * 2
        stft_scale = sep_src_stft_MFN.shape[1] - 1
        sep_src_stft_MFN /= stft_scale
        _, sep_src_sig_MT = istft(
            sep_src_stft_MFN, window='hann', nperseg=stft_winlen,
            noverlap=3*stft_winlen//4, nfft=stft_winlen)
        sep_src_sig_TM = sep_src_sig_MT.T #[:mix_sig_TM.shape[0]]
        sf.write(save_fileName, sep_src_sig_TM, 16000)


    def save_parameter(self, fileName):
        param_list = [
            self.lambda_NFT, self.covarianceDiag_NFM, self.diagonalizer_FMM]
        if self.xp != np:
            param_list = [
                self.convert_to_NumpyArray(param) for param in param_list]
        pic.dump(param_list, open(fileName, "wb"))


    def load_parameter(self, fileName):
        param_list = pic.load(open(fileName, "rb"))
        param_list = [self.xp.asarray(param) for param in param_list]
        self.lambda_NFT, self.covarianceDiag_NFM, self.diagonalizer_FMM =\
            param_list
        self.NUM_source, self.NUM_freq, self.NUM_time = self.lambda_NFT.shape
        self.NUM_mic = self.covarianceDiag_NFM.shape[-1]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'input_fileName', help='filename of the multichannel observed signals',
        type=str)
    parser.add_argument(
        '--file_id', help='file id',
        type=str, default="None")
    parser.add_argument(
        '--gpu', help='GPU ID',
        type=int, default=0)
    parser.add_argument(
        '--n_fft', help='number of frequencies',
        type=int, default=1024)
    parser.add_argument(
        '--NUM_source', help='number of noise',
        type=int, default=2)
    parser.add_argument(
        '--NUM_iteration', help='number of iteration',
        type=int, default=100)
    parser.add_argument(
        '--NUM_basis', help='number of basis',
        type=int, default=8)
    parser.add_argument(
        '--MODE_initialize_covarianceMatrix', help='unit, obs',
        type=str, default="obs")
    parser.add_argument(
        '--single_fp', action='store_true', dest="single_fp", default=False)
    args = parser.parse_args()

    if args.gpu < 0:
        xp = np
    else:
        import cupy as xp
        print("Use GPU " + str(args.gpu))
        xp.cuda.Device(args.gpu).use()

    if args.single_fp:
        C_FP_TYPE = xp.complex64
        F_FP_TYPE = xp.float32
    else:
        C_FP_TYPE = xp.complex128
        F_FP_TYPE = xp.float64

    sig, fs = sf.read(args.input_fileName, always_2d=True)
    spec_FNM = np.transpose(
        stft(sig.T, window='hann',
             nperseg=args.n_fft, noverlap=3*args.n_fft//4)[-1],
        (1, 2, 0))
    stft_scale = get_window('hann', args.n_fft).sum()
    spec_FNM *= stft_scale

    separater = FastFCA(
        NUM_source=args.NUM_source, xp=xp,
        MODE_initialize_covarianceMatrix=args.MODE_initialize_covarianceMatrix)
    separater.load_spectrogram(spec_FNM)
    separater.file_id = args.file_id
    separater.solve(
        NUM_iteration=args.NUM_iteration,
        save_likelihood=False, save_parameter=False, save_wav=False,
        save_path="./", interval_save_parameter=25)
