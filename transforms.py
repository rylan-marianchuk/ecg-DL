"""
transforms.py
Classes implementing common digital signal transformations:
Fourier
Short Time Fourier
PowerSpectrum
Wavelet
All transforms default signal time and sampling rate are set to T=10, fs=500 respectively, but remain modifiable
All transforms contain a __call__(signal) which is to be used in a pytorch dataset
    Signal is assumed to be of shape=(n,) of real valued floats
All transforms utilize torch tensors
All transforms contain a domain class variable and a domain_shape, outlining the tensor size of the resulting transform
    domain (dict), key 0: values of first axis, key 1: values of second axis (if 2D)
All transforms have a .view(signal) member function to visualize the transformed signal in plotly.graph_objects
@author Rylan Marianchuk
September 2021
"""
import plotly.graph_objs as go

import torch
import torch.nn.functional as F
import pywt
import math


class Fourier:
    def __init__(self, T=10, fs=500):
        """
        :param T: length of signal in seconds
        :param fs: (int) sampling rate of signal
        """
        self.fs = fs
        self.T = T
        self.domain = {0: torch.fft.rfftfreq(T*fs, 1/fs)}
        self.domain_shape = len(self.domain[0])
        return

    def __call__(self, signal):
        """
        Invoke numpy's real valued Fourier Transform
        :param signal: (float) shape=(n,) to transform
        :return: (complex) array of shape=domain_shape
        """
        return torch.fft.rfft(signal)

    def magnitude(self, signal):
        """
        Obtain the magnitude of the returned complex values of Fourier
        :param signal: (float) shape=(n,) to transform
        :return: (float64) shape=domain_shape magnitude of all complex components
        """
        complex = self(signal)
        return torch.abs(complex)

    def phase(self, signal):
        """
        Obtain the phase (angle) of the returned complex values of Fourier
        :param signal: (float) shape=(n,) to transform
        :return: (float64) shape=domain_shape phase of all complex components
        """
        complex = self(signal)
        return torch.angle(complex)

    def viewComplex(self, signal):
        """
        Generate plotly graph object to visualize the transform
        :param signal: (float) shape=(n,) to transform
        :return: viewable 2D complex plane of the Fourier transform
        """
        trfm = self(signal)
        fig = go.Figure(go.Scatter(x=torch.real(trfm), y=torch.imag(trfm), mode='markers'))
        fig.update_layout(title="Complex Plane of Fourier Transform")
        fig.update_xaxes(title_text="Real")
        fig.update_yaxes(title_text="Imaginary")
        fig.show()

    def viewMagnitude(self, signal):
        """
        Generate plotly graph object to visualize the transform
        :param signal: (float) shape=(n,) to transform
        :return: viewable violin plot of the magnitude of all complex components
        """
        fig = go.Figure(go.Violin(y=self.magnitude(signal), box_visible=True, points="all", name="Distribution Shape"))
        fig.update_layout(title="Fourier Magnitude Distribution")
        fig.update_yaxes(title_text="Magnitude")
        fig.show()

    def viewPhase(self, signal):
        """
        Generate plotly graph object to visualize the transform
        :param signal: (float) shape=(n,) to transform
        :return: viewable violin plot of the phase of all complex components
        """
        fig = go.Figure(go.Violin(y=self.phase(signal), box_visible=True, points="all", name="Distribution Shape"))
        fig.update_layout(title="Fourier Phase Distribution")
        fig.update_yaxes(title_text="Phase")
        fig.show()


class stFourier:
    def __init__(self, window_size, desired_windows, maxfreq_allowed=None, T=10, fs=500, log=False, normalize=False):
        """
        :param window_size: (int) number of signal elements to use in transform
        :param desired_windows: (int) amount of fourier windows to take across the signal, solves jump and finds n closest
        :param desired_windows: (int) maximum frequency provided in the image
        :param T: length of signal in seconds
        :param fs: (int) sampling rate of signal
        :param log: (bool) whether to take the logarithm of the resulting power
        :param normalize: (bool) whether to scale all results between 0 and 1, preserving the distances
        """
        assert (window_size < T*fs), "Specified window size greater than the signal length itself"
        self.fs = fs
        self.T = T
        self.maxfreq_allowed = maxfreq_allowed
        self.log = log
        self.normalize = normalize
        self.win = window_size
        # Solve for jump size so that no padding is used
        # Holding a possible jump size, and number of windows needed given that jump
        # D[0] - no remainder, D[1] will use 1 zero for padding, D[2] will use 2 zeros for padding
        D = {
            0: [],
            1: [],
            2: []
        }

        low_n = int(math.ceil(T * fs / self.win - 1 / self.win))
        high_n = int(T * fs - self.win)
        assert (low_n <= desired_windows <= high_n), "Desired number of windows is infeasible for signal"
        signal_len = int(T * fs)
        # Populate the dictionary
        for n in range(low_n, high_n):
            if (signal_len - 1 - self.win) % (n - 1) == 0:
                D[0].append((int((signal_len - 1 - self.win) / (n - 1)), n))
            elif (signal_len - 1 - self.win) % (n - 1) == 1:
                D[1].append((int((signal_len - 1 - self.win) / (n - 1)), n))
            elif (signal_len - 1 - self.win) % (n - 1) == 2:
                D[2].append((int((signal_len - 1 - self.win) / (n - 1)), n))

        # Get the remainder that contains jumps
        r = 0
        while len(D[r]) == 0:
            r += 1
            if r == 2: break

        self.jump, self.n_windows, diff = sorted([(pair[0], pair[1], abs(pair[1] - desired_windows)) for pair in D[r]], key=lambda x: x[2])[0]

        self.domain = {0: torch.tensor(range(0, T*fs-window_size, self.jump)) / fs,
                       1: torch.fft.rfftfreq(self.win, 1/fs)}
        self.freq_resolution = (self.domain[1][2] - self.domain[1][1]).item()
        self.domain_shape = (len(self.domain[0]), len(self.domain[1]))
        if maxfreq_allowed is None or maxfreq_allowed > fs / 2:
            self.image_shape = self.domain_shape
        else:
            self.image_shape = (len(self.domain[0]), int(maxfreq_allowed / self.freq_resolution))
        self.hamming = torch.hamming_window(self.win)

    def __call__(self, signal):
        """
        Slide window across signal taking the Fourier Transform at each
        :param signal: (float) shape=(n,) to transform
        :return: (float) shape=domain_shape 2D image of the stFT, each row a Power Spectrum at a given window start time
        """
        signal = signal.flatten()
        assert (signal.shape[0] == self.T*self.fs), "The signal is not corresponding to the specified time length and" \
                                                    "sample frequency"
        result = torch.zeros(self.domain_shape)
        signal_pad = torch.concat((signal, torch.zeros(self.win)))
        for i, L_edge in enumerate((range(0, signal.shape[0]-self.win, self.jump))):
            dampened = self.hamming * signal_pad[L_edge: L_edge + self.win]
            F = torch.fft.rfft(dampened)
            result[i] = torch.pow(torch.abs(F), 2)
            if self.log: result[i] = torch.log(result[i])

        if self.normalize:
            result -= torch.min(result)
            result /= torch.max(result) - torch.min(result)
            return result[:, :self.image_shape[1]].unsqueeze(0)
        return result[:, :self.image_shape[1]].unsqueeze(0)

    def view(self, signal, target=None):
        """
        Generate plotly graph object to visualize the transform
        :param signal: (float) shape=(n,) to transform
        :return: viewable Heatmap plot of the power at a given window and frequency
        """
        trfm = self(signal)
        fig = go.Figure(data=go.Heatmap(z=trfm[0], x=self.domain[1][:self.image_shape[1]], y=self.domain[0]))
        fig.update_layout(title="Short Time Fourier Transform  -  Window size: " + str(self.win) + "  -  Image Shape=" + str(self.image_shape) + "  -  Target: " + str(target))
        fig.update_yaxes(title_text="Window start time (seconds)", type='category')
        fig.update_xaxes(title_text="Power Spectrum of Window (frequency in Hz)", type='category')
        fig.show()


class PowerSpec:
    def __init__(self, T=10, fs=500, log=False, normalize=False):
        """
        :param T: length of signal in seconds
        :param fs: (int) sampling rate of signal
        :param log: (bool) whether to take the logarithm of the resulting power
        :param normalize: (bool) whether to scale all results between 0 and 1, preserving the distances
        """
        self.fs = fs
        self.T = T
        self.log = log
        self.normalize = normalize
        self.FT = Fourier(T=T, fs=fs)
        self.domain = {0: torch.fft.rfftfreq(T*fs, 1/fs)}
        self.domain_shape = len(self.domain[0])
        return

    def __call__(self, signal):
        """
        Obtain the power spectrum of the signal
        :param signal: (float) shape=(n,) to transform
        :return: (float) shape=domain_shape 1D Power Spectrum (magnitude of the FT squared)
        """
        signal = signal.flatten()
        assert (signal.shape[0] == self.T*self.fs), "The signal is not corresponding to the specified time length and" \
                                                    "sample frequency"
        if self.log:
            return torch.log(torch.pow(self.FT.magnitude(signal), 2))
        powerSpec = torch.pow(self.FT.magnitude(signal), 2)
        if self.normalize:
            powerSpec -= torch.min(powerSpec)
            powerSpec /= torch.max(powerSpec) - torch.min(powerSpec)
        return powerSpec

    def view(self, signal):
        """
        Generate plotly graph object to visualize the transform
        :param signal: (float) shape=(n,) to transform
        :return: viewable scatter plot of the Power of all frequencies in signal
        """
        trfm = self(signal)
        fig = go.Figure(go.Scatter(x=self.domain[0], y=trfm))
        fig.update_layout(title="PowerSpectrum Transform")
        fig.update_xaxes(title_text="Frequency (H)")
        fig.update_yaxes(title_text="Power  -  log=" + str(self.log))
        fig.show()


class Wavelet:
    def __init__(self, widths, output_size=None, wavelet='mexh', T=10, fs=500, normalize=False):
        """
        :param widths: (1D array) sequence of increasing wavelet widths, used as scale axis
        :param wavelet: (str) the chosen wavelet type. Options may be viewed with .seeAvailableWavelets()
        :param T: length of signal in seconds
        :param fs: (int) sampling rate of signal
        :param normalize: (bool) whether to scale all results between 0 and 1, preserving the distances
        """
        self.fs = fs
        self.T = T
        self.normalize = normalize
        self.wavelet = wavelet
        self.widths = widths
        self.n_widths = len(widths)
        self.domain_shape = (1, len(widths), T*fs)
        self.output_size = output_size
        if output_size is not None:
            self.domain_shape = (1, output_size[0], output_size[1])

        self.domain = { 0 : torch.linspace(widths.min(), widths.max(), self.domain_shape[1]), # each row a wavelet width is selected and translated across time
                        1: torch.linspace(0, T, self.domain_shape[2])} # linear time domain sequence
        return

    def __call__(self, ecg):
        """
        Transform the signal to its image of wavelet coefficients
        :param ecg: (tensor) shape=(n_leads,5000) to transform
        :return: (tensor) shape= 2D image of wavelet coefs
        """
        if len(ecg.shape) == 1:
            ecg = ecg.reshape(1, ecg.shape[0])
        trfm = pywt.cwt(ecg.numpy(), self.widths.numpy(), self.wavelet)[0]
        trfm = torch.from_numpy(trfm).transpose(0, 1)
        trfm = trfm.unsqueeze(1)
        if self.output_size is not None:
            trfm = F.interpolate(trfm, size=self.output_size)

        if self.normalize:
            for i in range(ecg.shape[0]):
                trfm[i] -= torch.min(trfm[i])
                trfm[i] /= torch.max(trfm[i]) - torch.min(trfm[i])

        return trfm

    def seeAvailableWavelets(self):
        print(pywt.wavelist(kind='continuous'))

    def view(self, ecg, index=0):
        """
        Generate plotly graph object to visualize the transform
        :param ecg: (tensor) shape=(n_leads,5000) to transform
        :param index:
        :return: viewable Heatmap plot viewing all wavelet coefficients at a given scale and translation time
        """
        trfm = self(ecg)
        fig = go.Figure(data=go.Heatmap(z=trfm[index][0], x=self.domain[1], y=self.domain[0].flip(0)))
        fig.update_layout(title="Wavelet Transform  -  Wavelet: " + str(self.wavelet))
        fig.update_yaxes(title_text="Wavelet scale", type='category')
        fig.update_xaxes(title_text="Time (seconds)", type='category')
        fig.update_yaxes(
            scaleanchor="x",
            scaleratio=1
        )
        fig.show()


class binaryImage:
    def __init__(self, resolution, max, min, T=10, fs=500):
        """
        :param resolution: (tuple) shape of desired binary image
            For now, x axis not scalable
        :param T: length of signal in seconds
        :param fs: (int) sampling rate of signal
        """
        self.fs = fs
        self.T = T
        self.min = min
        self.max = max
        self.domain = { 0 : torch.linspace(min, max, resolution[0]), # each row a wavelet width is selected and translated across time
                        1: torch.linspace(0, T, T*fs)} # linear time domain sequence
        self.domain_shape = (len(self.domain[0]), T*fs)
        return

    def __call__(self, signal):
        """
        Transform the signal to its 2D binary image curve
        :param signal: (float) shape=(n,) to transform
        :return: (float) shape=domain_shape 2D binary image of the signal
        """
        assert (signal.shape[0] == self.T*self.fs), "The signal is not corresponding to the specified time length and" \
                                                    "sample frequency"
        result = torch.zeros(shape=self.domain_shape)
        s_min = torch.min(signal)
        s_max = torch.max(signal)
        signal -= s_min
        signal /= s_max - s_min
        signal *= self.domain_shape[0] - 1
        signal = torch.rint(signal)
        for i in range(self.domain_shape[1]):
            result[int(signal[i]),i] = 1.0
        return result


    def view(self, signal):
        """
        Generate plotly graph object to visualize the transform
        :param signal: (float) shape=(n,) to transform
        :return: viewable Heatmap plot viewing all wavelet coefficients at a given scale and translation time
        """
        trfm = self(signal)
        fig = go.Figure(data=go.Heatmap(z=trfm, x=self.domain[1], y=self.domain[0]))
        fig.update_layout(title="Binarized Image")
        fig.update_yaxes(title_text="Wavelet scale", type='category')
        fig.update_xaxes(title_text="Time (seconds)", type='category')
        fig.show()

class GramianAngularField:
    def __init__(self, T=10, fs=500):
        self.T = T
        self.fs = fs

    def __call__(self):
        return


class MarkovTransitionField:
    def __init__(self, T=10, fs=500):
        self.T = T
        self.fs = fs

    def __call__(self):
        return


# ---------------- Inverse Transforms ----------------

class invFourier(object):
    def __init__(self, T=10, fs=500):
        self.fs = fs
        self.T = T
        self.domain = {0: torch.linspace(0, T, T*fs)}
        self.domain_shape = T*fs
        return

    def __call__(self, signal):
        return torch.fft.irfft(signal)
