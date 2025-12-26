"""
Module: MIMO Rayleigh Fading Channel với AWGN
===============================================
Mô hình kênh 2×2 MIMO với:
- Rayleigh fading (flat hoặc frequency-selective)
- AWGN (Additive White Gaussian Noise)

Mô hình kênh MIMO:
    y = H·x + n

Trong đó:
- y: Vector tín hiệu nhận (Nr × 1)
- H: Ma trận kênh (Nr × Nt)
- x: Vector tín hiệu phát (Nt × 1)
- n: Vector nhiễu AWGN (Nr × 1)

Rayleigh Fading:
- Mô hình hóa đường truyền không có LOS (Line of Sight)
- h_ij ~ CN(0, 1): Complex Gaussian với variance = 1
- |h_ij| có phân bố Rayleigh
"""

import numpy as np
from typing import Tuple, Optional


class RayleighMIMOChannel:
    """
    Kênh MIMO 2×2 Rayleigh Fading với AWGN.

    Mô hình: y = H·x + n

    Attributes:
        n_tx (int): Số anten phát
        n_rx (int): Số anten thu
        channel_type (str): 'flat' hoặc 'frequency_selective'
    """

    def __init__(self, n_tx: int = 2, n_rx: int = 2,
                 channel_type: str = 'flat',
                 n_taps: int = 1,
                 normalize: bool = True):
        """
        Khởi tạo kênh MIMO Rayleigh.

        Args:
            n_tx: Số anten phát
            n_rx: Số anten thu
            channel_type: 'flat' (1 tap) hoặc 'frequency_selective' (multi-tap)
            n_taps: Số taps cho frequency-selective channel
            normalize: Chuẩn hóa công suất kênh
        """
        self.n_tx = n_tx
        self.n_rx = n_rx
        self.channel_type = channel_type
        self.n_taps = n_taps if channel_type == 'frequency_selective' else 1
        self.normalize = normalize

        self.H = None  # Ma trận kênh hiện tại

    def generate_channel(self, n_subcarriers: int = 1) -> np.ndarray:
        """
        Tạo ma trận kênh Rayleigh mới.

        Flat fading: H[i,j] ~ CN(0, 1)
        Frequency-selective: H khác nhau cho mỗi subcarrier

        Args:
            n_subcarriers: Số subcarriers (cho OFDM)

        Returns:
            H: Ma trận kênh
               - Flat: (n_rx, n_tx)
               - Freq-selective: (n_subcarriers, n_rx, n_tx)
        """
        if self.channel_type == 'flat':
            # Complex Gaussian: CN(0, 1)
            H_real = np.random.randn(self.n_rx, self.n_tx)
            H_imag = np.random.randn(self.n_rx, self.n_tx)
            H = (H_real + 1j * H_imag) / np.sqrt(2)

            if self.normalize:
                # Chuẩn hóa để E[||H||^2_F] = n_rx * n_tx
                H = H / np.sqrt(np.mean(np.abs(H) ** 2)) * np.sqrt(1 / self.n_tx)

        else:  # frequency_selective
            # Tạo channel impulse response
            H_time = np.zeros((self.n_taps, self.n_rx, self.n_tx), dtype=complex)
            tap_powers = np.exp(-np.arange(self.n_taps) / 2)  # Exponential decay
            tap_powers = tap_powers / np.sum(tap_powers)  # Normalize

            for tap in range(self.n_taps):
                H_real = np.random.randn(self.n_rx, self.n_tx)
                H_imag = np.random.randn(self.n_rx, self.n_tx)
                H_time[tap] = np.sqrt(tap_powers[tap] / 2) * (H_real + 1j * H_imag)

            # FFT để chuyển sang frequency domain
            H = np.fft.fft(H_time, n=n_subcarriers, axis=0)

        self.H = H
        return H

    def apply_channel(self, tx_signal: np.ndarray,
                      snr_db: float,
                      H: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Áp dụng kênh MIMO và thêm nhiễu AWGN.

        y = H·x + n

        SNR được định nghĩa: SNR = E[||Hx||^2] / E[||n||^2]

        Args:
            tx_signal: Tín hiệu phát (n_tx × n_samples)
            snr_db: SNR tính bằng dB
            H: Ma trận kênh (optional, sử dụng self.H nếu None)

        Returns:
            rx_signal: Tín hiệu nhận được (n_rx × n_samples)
            noise: Nhiễu đã thêm
        """
        if H is None:
            if self.H is None:
                H = self.generate_channel()
            else:
                H = self.H

        n_samples = tx_signal.shape[1]

        # Áp dụng kênh
        if len(H.shape) == 2:  # Flat fading
            # y = H @ x
            rx_signal = H @ tx_signal
        else:  # Frequency selective (per-subcarrier)
            # Áp dụng kênh cho từng subcarrier
            rx_signal = np.zeros((self.n_rx, n_samples), dtype=complex)
            for k in range(min(H.shape[0], n_samples)):
                rx_signal[:, k] = H[k] @ tx_signal[:, k]

        # Tính công suất tín hiệu sau kênh
        signal_power = np.mean(np.abs(rx_signal) ** 2)

        # Tính công suất nhiễu theo SNR
        snr_linear = 10 ** (snr_db / 10)
        noise_power = signal_power / snr_linear

        # Tạo nhiễu AWGN phức
        noise_real = np.random.randn(self.n_rx, n_samples)
        noise_imag = np.random.randn(self.n_rx, n_samples)
        noise = np.sqrt(noise_power / 2) * (noise_real + 1j * noise_imag)

        rx_signal = rx_signal + noise

        return rx_signal, noise

    def get_noise_variance(self, snr_db: float, signal_power: float = 1.0) -> float:
        """
        Tính phương sai nhiễu từ SNR.

        Args:
            snr_db: SNR in dB
            signal_power: Công suất tín hiệu

        Returns:
            Phương sai nhiễu
        """
        snr_linear = 10 ** (snr_db / 10)
        return signal_power / snr_linear


class FrequencyFlatMIMOChannel:
    """
    Kênh MIMO-OFDM với flat fading theo block.

    Kênh được giả định không đổi trong một OFDM symbol
    nhưng có thể thay đổi giữa các symbols.

    Phù hợp cho mô phỏng MIMO-OFDM với:
    - Block fading model
    - Channel estimation per symbol
    """

    def __init__(self, n_tx: int = 2, n_rx: int = 2):
        """
        Khởi tạo kênh.

        Args:
            n_tx: Số anten phát
            n_rx: Số anten thu
        """
        self.n_tx = n_tx
        self.n_rx = n_rx

    def generate_channel_matrix(self, n_subcarriers: int = 64) -> np.ndarray:
        """
        Tạo ma trận kênh cho tất cả subcarriers.

        Với flat fading, H giống nhau cho mọi subcarrier
        trong cùng một OFDM symbol.

        Args:
            n_subcarriers: Số subcarriers

        Returns:
            H: Ma trận kênh (n_subcarriers, n_rx, n_tx)
        """
        # Flat fading: cùng H cho mọi subcarrier
        H_real = np.random.randn(self.n_rx, self.n_tx)
        H_imag = np.random.randn(self.n_rx, self.n_tx)
        H_flat = (H_real + 1j * H_imag) / np.sqrt(2)

        # Nhân bản cho tất cả subcarriers
        H = np.tile(H_flat, (n_subcarriers, 1, 1))

        return H

    def generate_frequency_selective(self, n_subcarriers: int = 64,
                                      n_taps: int = 4) -> np.ndarray:
        """
        Tạo kênh frequency-selective.

        Mô hình: H(k) = sum_l h_l * exp(-j*2*pi*k*l/N)

        Args:
            n_subcarriers: Số subcarriers
            n_taps: Số taps (độ dài channel impulse response)

        Returns:
            H: Ma trận kênh (n_subcarriers, n_rx, n_tx)
        """
        # Power delay profile (exponential decay)
        tap_powers = np.exp(-np.arange(n_taps) * 0.5)
        tap_powers = tap_powers / np.sum(tap_powers)

        # Channel impulse response
        h_time = np.zeros((n_taps, self.n_rx, self.n_tx), dtype=complex)

        for l in range(n_taps):
            h_real = np.random.randn(self.n_rx, self.n_tx)
            h_imag = np.random.randn(self.n_rx, self.n_tx)
            h_time[l] = np.sqrt(tap_powers[l] / 2) * (h_real + 1j * h_imag)

        # FFT để có channel frequency response
        # H(k) for k = 0, 1, ..., N-1
        H = np.zeros((n_subcarriers, self.n_rx, self.n_tx), dtype=complex)

        for k in range(n_subcarriers):
            for l in range(n_taps):
                phase = np.exp(-2j * np.pi * k * l / n_subcarriers)
                H[k] += h_time[l] * phase

        return H

    def apply_channel_ofdm(self, tx_freq: np.ndarray,
                           H: np.ndarray,
                           snr_db: float) -> Tuple[np.ndarray, float]:
        """
        Áp dụng kênh cho tín hiệu OFDM trong miền tần số.

        y[k] = H[k] @ x[k] + n[k]  cho mỗi subcarrier k

        Args:
            tx_freq: Tín hiệu phát miền tần số (n_tx, n_subcarriers)
            H: Ma trận kênh (n_subcarriers, n_rx, n_tx)
            snr_db: SNR in dB

        Returns:
            rx_freq: Tín hiệu nhận miền tần số (n_rx, n_subcarriers)
            noise_var: Phương sai nhiễu
        """
        n_subcarriers = tx_freq.shape[1]
        rx_freq = np.zeros((self.n_rx, n_subcarriers), dtype=complex)

        # Áp dụng kênh cho từng subcarrier
        for k in range(n_subcarriers):
            rx_freq[:, k] = H[k] @ tx_freq[:, k]

        # Tính công suất tín hiệu
        signal_power = np.mean(np.abs(rx_freq) ** 2)

        # Thêm nhiễu
        snr_linear = 10 ** (snr_db / 10)
        noise_power = signal_power / snr_linear

        noise_real = np.random.randn(self.n_rx, n_subcarriers)
        noise_imag = np.random.randn(self.n_rx, n_subcarriers)
        noise = np.sqrt(noise_power / 2) * (noise_real + 1j * noise_imag)

        rx_freq = rx_freq + noise

        return rx_freq, noise_power


def theoretical_ber_rayleigh(snr_db: np.ndarray, modulation: str = 'bpsk') -> np.ndarray:
    """
    Tính BER lý thuyết cho kênh Rayleigh.

    Công thức cho BPSK/QPSK:
    BER = 0.5 * (1 - sqrt(SNR / (1 + SNR)))

    Args:
        snr_db: Array SNR values in dB
        modulation: Kiểu điều chế

    Returns:
        BER lý thuyết
    """
    snr_linear = 10 ** (snr_db / 10)

    if modulation.lower() in ['bpsk', 'qpsk']:
        # Rayleigh BER for BPSK/QPSK
        ber = 0.5 * (1 - np.sqrt(snr_linear / (1 + snr_linear)))
    elif modulation.lower() == '16qam':
        # Approximate for 16-QAM
        ber = 0.2 * (1 - np.sqrt(snr_linear / (10 + snr_linear)))
    elif modulation.lower() == '64qam':
        # Approximate for 64-QAM
        ber = 7 / 24 * (1 - np.sqrt(snr_linear / (42 + snr_linear)))
    else:
        raise ValueError(f"Unknown modulation: {modulation}")

    return ber


def theoretical_ber_awgn(snr_db: np.ndarray, modulation: str = 'bpsk') -> np.ndarray:
    """
    Tính BER lý thuyết cho kênh AWGN.

    BPSK/QPSK: BER = Q(sqrt(2*SNR))

    Args:
        snr_db: Array SNR values in dB
        modulation: Kiểu điều chế

    Returns:
        BER lý thuyết
    """
    from scipy.special import erfc

    snr_linear = 10 ** (snr_db / 10)

    if modulation.lower() in ['bpsk', 'qpsk']:
        ber = 0.5 * erfc(np.sqrt(snr_linear))
    elif modulation.lower() == '16qam':
        # Approximate
        ber = 0.75 * 0.5 * erfc(np.sqrt(snr_linear / 10))
    elif modulation.lower() == '64qam':
        # Approximate
        ber = 7 / 12 * 0.5 * erfc(np.sqrt(snr_linear / 42))
    else:
        raise ValueError(f"Unknown modulation: {modulation}")

    return ber


# Test module
if __name__ == "__main__":
    print("=" * 60)
    print("TEST MODULE: MIMO Rayleigh Channel")
    print("=" * 60)

    np.random.seed(42)

    # Test 2x2 MIMO channel
    print("\n--- Test 2×2 MIMO Rayleigh Channel ---")
    channel = RayleighMIMOChannel(n_tx=2, n_rx=2, channel_type='flat')

    # Generate channel
    H = channel.generate_channel()
    print(f"\nChannel matrix H (2×2):")
    print(f"  H[0,0] = {H[0,0]:.4f}")
    print(f"  H[0,1] = {H[0,1]:.4f}")
    print(f"  H[1,0] = {H[1,0]:.4f}")
    print(f"  H[1,1] = {H[1,1]:.4f}")

    # Check Rayleigh distribution
    n_samples = 10000
    h_samples = []
    for _ in range(n_samples):
        H = channel.generate_channel()
        h_samples.append(np.abs(H[0, 0]))

    h_samples = np.array(h_samples)
    print(f"\n|h| statistics (should follow Rayleigh):")
    print(f"  Mean: {np.mean(h_samples):.4f} (theoretical: {np.sqrt(np.pi/2)/np.sqrt(2):.4f})")
    print(f"  Variance: {np.var(h_samples):.4f}")

    # Test channel application
    print("\n--- Test Channel Application ---")
    H = channel.generate_channel()
    tx_signal = np.array([[1 + 1j, -1 + 1j], [-1 - 1j, 1 - 1j]]) / np.sqrt(2)
    print(f"TX signal shape: {tx_signal.shape}")

    for snr_db in [0, 10, 20]:
        rx_signal, noise = channel.apply_channel(tx_signal, snr_db, H)
        actual_snr = 10 * np.log10(np.mean(np.abs(H @ tx_signal) ** 2) /
                                    np.mean(np.abs(noise) ** 2))
        print(f"Target SNR: {snr_db} dB, Actual SNR: {actual_snr:.2f} dB")

    # Test frequency-selective channel
    print("\n--- Test Frequency-Selective Channel ---")
    ofdm_channel = FrequencyFlatMIMOChannel(n_tx=2, n_rx=2)
    H_freq = ofdm_channel.generate_frequency_selective(n_subcarriers=64, n_taps=4)
    print(f"Channel shape: {H_freq.shape}")
    print(f"Channel varies across subcarriers: {np.std(np.abs(H_freq[:, 0, 0])):.4f}")

    print("\n" + "=" * 60)
    print("Test PASSED!")
    print("=" * 60)
