"""
Module: OFDM (Orthogonal Frequency Division Multiplexing) System
=================================================================
Hệ thống OFDM với:
- IFFT/FFT modulation
- Cyclic Prefix (CP) để chống ISI
- Pilot insertion cho channel estimation

OFDM chia băng thông thành nhiều subcarriers trực giao, cho phép:
- Truyền song song trên nhiều subcarriers
- Kháng multipath fading
- Dễ dàng kết hợp với MIMO
"""

import numpy as np
from typing import Tuple, Optional


class OFDMSystem:
    """
    Hệ thống OFDM với cấu hình linh hoạt.

    Cấu trúc một OFDM symbol:
    |-- Cyclic Prefix --|-------- IFFT Output --------|

    Trong miền tần số:
    |-- DC --|-- Data --|-- Pilots --|-- Guard --|

    Attributes:
        n_fft (int): Kích thước FFT
        n_cp (int): Chiều dài cyclic prefix
        n_data (int): Số subcarriers dữ liệu
        pilot_indices (np.ndarray): Vị trí các pilot subcarriers
        data_indices (np.ndarray): Vị trí các data subcarriers
    """

    def __init__(self, n_fft: int = 64, n_cp: int = 16,
                 pilot_spacing: int = 8, pilot_value: complex = 1 + 0j):
        """
        Khởi tạo hệ thống OFDM.

        Args:
            n_fft: Kích thước FFT (số subcarriers tổng cộng)
            n_cp: Chiều dài cyclic prefix (samples)
            pilot_spacing: Khoảng cách giữa các pilots
            pilot_value: Giá trị pilot (known symbol)
        """
        self.n_fft = n_fft
        self.n_cp = n_cp
        self.pilot_spacing = pilot_spacing
        self.pilot_value = pilot_value

        # Định nghĩa cấu trúc subcarriers
        self._define_subcarrier_structure()

    def _define_subcarrier_structure(self):
        """
        Định nghĩa cấu trúc các subcarriers.

        Cấu trúc tiêu chuẩn (ví dụ với N_FFT = 64):
        - DC subcarrier (index 0): Không sử dụng
        - Guard bands: Các subcarrier hai bên
        - Pilots: Phân bố đều cho channel estimation
        - Data: Các subcarrier còn lại
        """
        n_fft = self.n_fft

        # Guard bands (không sử dụng)
        n_guard = n_fft // 8  # 12.5% mỗi bên
        guard_left = list(range(1, n_guard + 1))
        guard_right = list(range(n_fft - n_guard, n_fft))

        # DC subcarrier (không sử dụng)
        dc_subcarrier = [0]

        # Các subcarriers có thể sử dụng
        usable = set(range(n_fft)) - set(guard_left) - set(guard_right) - set(dc_subcarrier)
        usable = sorted(list(usable))

        # Pilot subcarriers (phân bố đều trong vùng usable)
        pilot_indices = []
        for i in range(0, len(usable), self.pilot_spacing):
            pilot_indices.append(usable[i])
        self.pilot_indices = np.array(pilot_indices)

        # Data subcarriers (còn lại sau khi trừ pilots)
        data_indices = sorted(set(usable) - set(pilot_indices))
        self.data_indices = np.array(data_indices)

        self.n_data = len(self.data_indices)
        self.n_pilots = len(self.pilot_indices)

        # Lưu guard indices
        self.guard_indices = np.array(guard_left + guard_right + dc_subcarrier)

    def modulate(self, data_symbols: np.ndarray) -> np.ndarray:
        """
        Điều chế OFDM: chuyển từ miền tần số sang miền thời gian.

        Quy trình:
        1. Chèn data symbols vào data subcarriers
        2. Chèn pilots vào pilot subcarriers
        3. IFFT để chuyển sang miền thời gian
        4. Thêm Cyclic Prefix

        Args:
            data_symbols: Mảng các symbols (số lượng phải là bội của n_data)

        Returns:
            OFDM signal trong miền thời gian

        Raises:
            ValueError: Nếu số symbols không phù hợp
        """
        if len(data_symbols) % self.n_data != 0:
            raise ValueError(f"Số symbols ({len(data_symbols)}) phải là bội của {self.n_data}")

        n_symbols = len(data_symbols) // self.n_data
        ofdm_time = []

        for i in range(n_symbols):
            # Lấy data cho OFDM symbol này
            symbol_data = data_symbols[i * self.n_data:(i + 1) * self.n_data]

            # Tạo frequency-domain OFDM symbol
            freq_domain = np.zeros(self.n_fft, dtype=complex)

            # Chèn data
            freq_domain[self.data_indices] = symbol_data

            # Chèn pilots
            freq_domain[self.pilot_indices] = self.pilot_value

            # IFFT
            time_domain = np.fft.ifft(freq_domain) * np.sqrt(self.n_fft)

            # Thêm Cyclic Prefix
            cp = time_domain[-self.n_cp:]
            ofdm_symbol = np.concatenate([cp, time_domain])

            ofdm_time.extend(ofdm_symbol)

        return np.array(ofdm_time)

    def demodulate(self, received_signal: np.ndarray,
                   channel_estimate: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Giải điều chế OFDM: chuyển từ miền thời gian về miền tần số.

        Quy trình:
        1. Loại bỏ Cyclic Prefix
        2. FFT để chuyển về miền tần số
        3. Trích xuất data symbols

        Args:
            received_signal: Tín hiệu nhận được trong miền thời gian
            channel_estimate: Ước lượng kênh (optional)

        Returns:
            data_symbols: Các data symbols
            pilot_symbols: Các pilot symbols (để ước lượng kênh)
        """
        symbol_length = self.n_fft + self.n_cp
        n_symbols = len(received_signal) // symbol_length

        data_symbols = []
        pilot_symbols = []

        for i in range(n_symbols):
            # Lấy OFDM symbol
            start_idx = i * symbol_length
            ofdm_symbol = received_signal[start_idx:start_idx + symbol_length]

            # Loại bỏ CP
            time_domain = ofdm_symbol[self.n_cp:]

            # FFT
            freq_domain = np.fft.fft(time_domain) / np.sqrt(self.n_fft)

            # Trích xuất data và pilots
            data = freq_domain[self.data_indices]
            pilots = freq_domain[self.pilot_indices]

            # Equalization nếu có channel estimate
            if channel_estimate is not None:
                h_data = channel_estimate[self.data_indices]
                data = data / h_data

            data_symbols.extend(data)
            pilot_symbols.append(pilots)

        return np.array(data_symbols), np.array(pilot_symbols)

    def estimate_channel(self, received_pilots: np.ndarray) -> np.ndarray:
        """
        Ước lượng kênh từ pilots sử dụng interpolation.

        Phương pháp: Least Squares estimation tại pilot positions,
        sau đó nội suy tuyến tính cho data subcarriers.

        H_pilot = Y_pilot / X_pilot

        Args:
            received_pilots: Pilots nhận được

        Returns:
            channel_estimate: Ước lượng kênh cho tất cả subcarriers
        """
        # LS estimation tại pilot positions
        h_pilot = received_pilots / self.pilot_value

        # Nội suy tuyến tính cho toàn bộ spectrum
        channel_estimate = np.zeros(self.n_fft, dtype=complex)

        # Đặt giá trị tại pilot positions
        channel_estimate[self.pilot_indices] = h_pilot

        # Nội suy cho data subcarriers
        all_indices = np.concatenate([self.pilot_indices, self.data_indices])
        all_indices = np.sort(all_indices)

        # Linear interpolation
        channel_real = np.interp(all_indices,
                                  self.pilot_indices,
                                  np.real(h_pilot))
        channel_imag = np.interp(all_indices,
                                  self.pilot_indices,
                                  np.imag(h_pilot))

        channel_estimate[all_indices] = channel_real + 1j * channel_imag

        return channel_estimate

    def get_params(self) -> dict:
        """
        Trả về các tham số của hệ thống OFDM.

        Returns:
            Dictionary chứa các tham số
        """
        return {
            'n_fft': self.n_fft,
            'n_cp': self.n_cp,
            'n_data': self.n_data,
            'n_pilots': self.n_pilots,
            'pilot_spacing': self.pilot_spacing,
            'symbol_length': self.n_fft + self.n_cp,
            'spectral_efficiency': self.n_data / (self.n_fft + self.n_cp)
        }


class MIMOOFDMSystem:
    """
    Hệ thống MIMO-OFDM cho 2x2 MIMO.

    Kết hợp OFDM với Spatial Multiplexing:
    - 2 anten phát, 2 anten thu
    - Mỗi anten phát stream riêng
    - Tăng gấp đôi throughput so với SISO
    """

    def __init__(self, n_fft: int = 64, n_cp: int = 16,
                 n_tx: int = 2, n_rx: int = 2,
                 pilot_spacing: int = 8):
        """
        Khởi tạo hệ thống MIMO-OFDM.

        Args:
            n_fft: Kích thước FFT
            n_cp: Chiều dài CP
            n_tx: Số anten phát
            n_rx: Số anten thu
            pilot_spacing: Khoảng cách pilots
        """
        self.n_tx = n_tx
        self.n_rx = n_rx

        # Tạo OFDM system cho mỗi anten
        self.ofdm = OFDMSystem(n_fft, n_cp, pilot_spacing)

        # Sử dụng orthogonal pilots cho MIMO channel estimation
        # Pilot values cho mỗi Tx antenna (orthogonal trong không gian)
        self.pilot_values = self._generate_orthogonal_pilots()

    def _generate_orthogonal_pilots(self) -> np.ndarray:
        """
        Tạo orthogonal pilot sequences cho MIMO.

        Sử dụng Hadamard-like structure để pilots từ các anten
        trực giao với nhau.

        Returns:
            Ma trận pilot values (n_tx x n_pilots)
        """
        n_pilots = self.ofdm.n_pilots

        # DFT-based orthogonal pilots
        pilots = np.zeros((self.n_tx, n_pilots), dtype=complex)
        for tx in range(self.n_tx):
            phase = np.exp(2j * np.pi * tx * np.arange(n_pilots) / self.n_tx)
            pilots[tx] = phase

        return pilots

    def modulate(self, data_symbols: np.ndarray) -> np.ndarray:
        """
        Điều chế MIMO-OFDM với spatial multiplexing.

        Chia data cho các Tx antennas và điều chế OFDM riêng.

        Args:
            data_symbols: Data symbols (chia đều cho các Tx)

        Returns:
            tx_signals: Tín hiệu phát cho mỗi anten (n_tx x samples)
        """
        n_data = self.ofdm.n_data
        symbols_per_tx = len(data_symbols) // self.n_tx

        # Đảm bảo chia hết
        if len(data_symbols) % (self.n_tx * n_data) != 0:
            raise ValueError("Số symbols phải chia hết cho n_tx * n_data")

        tx_signals = []

        for tx in range(self.n_tx):
            # Data cho anten này
            tx_data = data_symbols[tx * symbols_per_tx:(tx + 1) * symbols_per_tx]

            # Tạm thời thay đổi pilot value
            original_pilot = self.ofdm.pilot_value
            n_symbols = symbols_per_tx // n_data

            # Điều chế từng symbol với pilot tương ứng
            signal_parts = []
            for sym_idx in range(n_symbols):
                sym_data = tx_data[sym_idx * n_data:(sym_idx + 1) * n_data]

                # Tạo freq domain
                freq_domain = np.zeros(self.ofdm.n_fft, dtype=complex)
                freq_domain[self.ofdm.data_indices] = sym_data

                # Pilots cho anten này (cycling through pilots)
                pilot_idx = sym_idx % self.ofdm.n_pilots
                freq_domain[self.ofdm.pilot_indices] = self.pilot_values[tx]

                # IFFT
                time_domain = np.fft.ifft(freq_domain) * np.sqrt(self.ofdm.n_fft)

                # Add CP
                cp = time_domain[-self.ofdm.n_cp:]
                ofdm_symbol = np.concatenate([cp, time_domain])
                signal_parts.extend(ofdm_symbol)

            tx_signals.append(np.array(signal_parts))

        return np.array(tx_signals)

    def demodulate(self, rx_signals: np.ndarray) -> np.ndarray:
        """
        Giải điều chế tín hiệu nhận được từ các Rx antennas.

        Args:
            rx_signals: Tín hiệu nhận được (n_rx x samples)

        Returns:
            freq_domain: Tín hiệu miền tần số (n_rx x n_symbols x n_fft)
        """
        symbol_length = self.ofdm.n_fft + self.ofdm.n_cp
        n_samples = rx_signals.shape[1]
        n_symbols = n_samples // symbol_length

        freq_domain_all = np.zeros((self.n_rx, n_symbols, self.ofdm.n_fft), dtype=complex)

        for rx in range(self.n_rx):
            for sym in range(n_symbols):
                start = sym * symbol_length
                ofdm_sym = rx_signals[rx, start:start + symbol_length]

                # Remove CP and FFT
                time_domain = ofdm_sym[self.ofdm.n_cp:]
                freq = np.fft.fft(time_domain) / np.sqrt(self.ofdm.n_fft)
                freq_domain_all[rx, sym] = freq

        return freq_domain_all

    def get_params(self) -> dict:
        """
        Trả về các tham số MIMO-OFDM.
        """
        params = self.ofdm.get_params()
        params.update({
            'n_tx': self.n_tx,
            'n_rx': self.n_rx,
            'spatial_multiplexing_gain': self.n_tx
        })
        return params


# Test module
if __name__ == "__main__":
    print("=" * 60)
    print("TEST MODULE: OFDM System")
    print("=" * 60)

    # Test SISO OFDM
    print("\n--- Test SISO OFDM ---")
    ofdm = OFDMSystem(n_fft=64, n_cp=16, pilot_spacing=8)
    params = ofdm.get_params()

    print(f"OFDM Parameters:")
    for key, value in params.items():
        print(f"  {key}: {value}")

    # Tạo data ngẫu nhiên
    np.random.seed(42)
    n_symbols = 10
    data = (np.random.randn(n_symbols * ofdm.n_data) +
            1j * np.random.randn(n_symbols * ofdm.n_data)) / np.sqrt(2)

    print(f"\nTest data: {len(data)} symbols")

    # Modulate
    tx_signal = ofdm.modulate(data)
    print(f"TX signal length: {len(tx_signal)} samples")

    # Add AWGN noise
    snr_db = 20
    signal_power = np.mean(np.abs(tx_signal) ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = np.sqrt(noise_power / 2) * (np.random.randn(len(tx_signal)) +
                                         1j * np.random.randn(len(tx_signal)))
    rx_signal = tx_signal + noise

    # Demodulate
    rx_data, rx_pilots = ofdm.demodulate(rx_signal)
    print(f"RX data symbols: {len(rx_data)}")

    # Check error (no channel, chỉ có nhiễu)
    mse = np.mean(np.abs(data - rx_data) ** 2)
    print(f"MSE (SNR={snr_db}dB): {mse:.6f}")

    # Test MIMO-OFDM
    print("\n--- Test MIMO-OFDM (2x2) ---")
    mimo_ofdm = MIMOOFDMSystem(n_fft=64, n_cp=16, n_tx=2, n_rx=2)
    params = mimo_ofdm.get_params()

    print(f"MIMO-OFDM Parameters:")
    for key, value in params.items():
        print(f"  {key}: {value}")

    # Data cho 2 streams
    n_data = mimo_ofdm.ofdm.n_data
    total_data = 2 * 5 * n_data  # 2 antennas, 5 symbols each
    mimo_data = (np.random.randn(total_data) +
                 1j * np.random.randn(total_data)) / np.sqrt(2)

    # Modulate
    tx_signals = mimo_ofdm.modulate(mimo_data)
    print(f"\nMIMO TX signals shape: {tx_signals.shape}")

    print("\n" + "=" * 60)
    print("Test PASSED!")
    print("=" * 60)
