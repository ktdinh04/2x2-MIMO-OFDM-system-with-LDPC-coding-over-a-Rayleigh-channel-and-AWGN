"""
2×2 MIMO-OFDM System with LDPC Coding Simulation
=================================================

Mô phỏng hoàn chỉnh hệ thống truyền thông không dây:
- 2×2 MIMO với Spatial Multiplexing
- OFDM với Cyclic Prefix
- Mã hóa LDPC rate 1/2
- Điều chế 64-QAM
- Kênh Rayleigh fading + AWGN
- Đánh giá BER và SER

Chuỗi xử lý:
TX: Bits → LDPC Encode → 64-QAM → OFDM Mod → MIMO TX
Channel: Rayleigh Fading + AWGN
RX: MIMO Detect → OFDM Demod → 64-QAM Demod → LDPC Decode → Bits
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from qam_modulation import QAM64
from ldpc_codec import LDPCCode, LDPCCodec
from ofdm_system import OFDMSystem
from mimo_channel import FrequencyFlatMIMOChannel
from mimo_detector import MIMODetector


class MIMOOFDMLDPCSystem:
    """
    Hệ thống 2×2 MIMO-OFDM với mã hóa LDPC hoàn chỉnh.
    """

    def __init__(self,
                 n_fft: int = 64,
                 n_cp: int = 16,
                 n_tx: int = 2,
                 n_rx: int = 2,
                 ldpc_n: int = 648,
                 ldpc_rate: float = 0.5,
                 seed: int = 42):
        """
        Khởi tạo hệ thống.

        Args:
            n_fft: Kích thước FFT cho OFDM
            n_cp: Chiều dài Cyclic Prefix
            n_tx: Số anten phát
            n_rx: Số anten thu
            ldpc_n: Chiều dài codeword LDPC
            ldpc_rate: Tỷ lệ mã hóa LDPC
            seed: Random seed
        """
        self.n_tx = n_tx
        self.n_rx = n_rx
        self.seed = seed

        # Khởi tạo các module
        print("Initializing system components...")

        # 64-QAM modulator
        self.qam = QAM64()
        self.bits_per_symbol = 6

        # OFDM system
        self.ofdm = OFDMSystem(n_fft=n_fft, n_cp=n_cp, pilot_spacing=8)
        self.n_data = self.ofdm.n_data

        # LDPC codec (với kích thước nhỏ hơn để tăng tốc)
        self.ldpc = LDPCCode(n=ldpc_n, rate=ldpc_rate, seed=seed)
        self.ldpc_k = self.ldpc.k

        # MIMO channel
        self.channel = FrequencyFlatMIMOChannel(n_tx=n_tx, n_rx=n_rx)

        # MIMO detector
        self.detector = MIMODetector(n_tx=n_tx, n_rx=n_rx)

        print(f"System initialized:")
        print(f"  - MIMO: {n_tx}×{n_rx}")
        print(f"  - OFDM: {n_fft} subcarriers, {n_cp} CP, {self.n_data} data subcarriers")
        print(f"  - LDPC: ({self.ldpc.n}, {self.ldpc.k}), rate = {ldpc_rate}")
        print(f"  - Modulation: 64-QAM ({self.bits_per_symbol} bits/symbol)")

    def transmit(self, info_bits: np.ndarray) -> tuple:
        """
        Chuỗi xử lý phía phát.

        Args:
            info_bits: Bit thông tin đầu vào

        Returns:
            tx_freq: Tín hiệu phát trong miền tần số
            tx_symbols: Symbols đã điều chế (để tính SER)
        """
        # 1. LDPC Encoding
        # Padding để chia hết cho ldpc_k
        n_bits = len(info_bits)
        n_blocks = int(np.ceil(n_bits / self.ldpc_k))
        padded_bits = np.zeros(n_blocks * self.ldpc_k, dtype=int)
        padded_bits[:n_bits] = info_bits

        coded_bits = np.zeros(n_blocks * self.ldpc.n, dtype=int)
        for i in range(n_blocks):
            msg = padded_bits[i * self.ldpc_k:(i + 1) * self.ldpc_k]
            coded_bits[i * self.ldpc.n:(i + 1) * self.ldpc.n] = self.ldpc.encode(msg)

        # 2. 64-QAM Modulation
        # Padding để chia hết cho 6 (bits per symbol)
        n_coded = len(coded_bits)
        n_symbols_needed = int(np.ceil(n_coded / self.bits_per_symbol))
        padded_coded = np.zeros(n_symbols_needed * self.bits_per_symbol, dtype=int)
        padded_coded[:n_coded] = coded_bits

        symbols = self.qam.modulate(padded_coded)

        # 3. Chia symbols cho các Tx antennas
        # Padding để chia đều cho n_tx và n_data
        symbols_per_ofdm = self.n_data * self.n_tx
        n_ofdm_symbols = int(np.ceil(len(symbols) / symbols_per_ofdm))
        total_symbols = n_ofdm_symbols * symbols_per_ofdm
        padded_symbols = np.zeros(total_symbols, dtype=complex)
        padded_symbols[:len(symbols)] = symbols

        # Reshape: (n_ofdm, n_tx, n_data) → (n_tx, n_ofdm * n_data)
        symbols_reshaped = padded_symbols.reshape(n_ofdm_symbols, self.n_tx, self.n_data)
        tx_symbols_per_ant = symbols_reshaped.transpose(1, 0, 2).reshape(self.n_tx, -1)

        # 4. OFDM Modulation (per antenna)
        tx_freq = np.zeros((self.n_tx, n_ofdm_symbols, self.ofdm.n_fft), dtype=complex)

        for tx in range(self.n_tx):
            for sym_idx in range(n_ofdm_symbols):
                freq_domain = np.zeros(self.ofdm.n_fft, dtype=complex)
                data = tx_symbols_per_ant[tx, sym_idx * self.n_data:(sym_idx + 1) * self.n_data]
                freq_domain[self.ofdm.data_indices] = data
                freq_domain[self.ofdm.pilot_indices] = 1.0  # Pilot = 1
                tx_freq[tx, sym_idx] = freq_domain

        return tx_freq, padded_symbols, n_bits, n_blocks

    def channel_propagation(self, tx_freq: np.ndarray, snr_db: float) -> tuple:
        """
        Truyền qua kênh MIMO Rayleigh + AWGN.

        Args:
            tx_freq: Tín hiệu phát (n_tx, n_ofdm_symbols, n_fft)
            snr_db: SNR in dB

        Returns:
            rx_freq: Tín hiệu nhận
            H_true: Ma trận kênh thực
            noise_var: Phương sai nhiễu
        """
        n_ofdm_symbols = tx_freq.shape[1]
        n_fft = tx_freq.shape[2]

        rx_freq = np.zeros((self.n_rx, n_ofdm_symbols, n_fft), dtype=complex)
        H_all = []

        snr_linear = 10 ** (snr_db / 10)

        for sym_idx in range(n_ofdm_symbols):
            # Generate channel for this OFDM symbol
            H = self.channel.generate_frequency_selective(n_subcarriers=n_fft, n_taps=4)
            H_all.append(H)

            # Apply channel per subcarrier
            for k in range(n_fft):
                tx_k = tx_freq[:, sym_idx, k]  # (n_tx,)
                rx_freq[:, sym_idx, k] = H[k] @ tx_k

        # Add AWGN
        signal_power = np.mean(np.abs(rx_freq) ** 2)
        noise_power = signal_power / snr_linear
        noise = np.sqrt(noise_power / 2) * (
            np.random.randn(*rx_freq.shape) + 1j * np.random.randn(*rx_freq.shape)
        )
        rx_freq = rx_freq + noise

        return rx_freq, H_all, noise_power

    def receive(self, rx_freq: np.ndarray, H_all: list, noise_var: float,
                n_info_bits: int, n_blocks: int,
                detection_method: str = 'mmse') -> tuple:
        """
        Chuỗi xử lý phía thu.

        Args:
            rx_freq: Tín hiệu nhận
            H_all: Danh sách ma trận kênh
            noise_var: Phương sai nhiễu
            n_info_bits: Số bit thông tin gốc
            n_blocks: Số block LDPC
            detection_method: 'zf' hoặc 'mmse'

        Returns:
            decoded_bits: Bit đã giải mã
            rx_symbols: Symbols đã nhận (sau detection)
        """
        n_ofdm_symbols = rx_freq.shape[1]
        n_fft = rx_freq.shape[2]

        # 1. MIMO Detection + OFDM Demodulation
        detected_symbols = []

        for sym_idx in range(n_ofdm_symbols):
            H = H_all[sym_idx]

            for k in self.ofdm.data_indices:
                y_k = rx_freq[:, sym_idx, k]
                H_k = H[k]

                if detection_method == 'zf':
                    x_hat = self.detector.detect_zf(y_k, H_k)
                else:  # mmse
                    x_hat = self.detector.detect_mmse(y_k, H_k, noise_var)

                detected_symbols.extend(x_hat)

        detected_symbols = np.array(detected_symbols)

        # 2. 64-QAM Soft Demodulation
        llr = self.qam.demodulate_soft(detected_symbols, noise_var)

        # 3. LDPC Decoding
        # Lấy đúng số bit cần thiết
        llr_needed = n_blocks * self.ldpc.n
        llr_for_decode = llr[:llr_needed]

        decoded_bits = np.zeros(n_blocks * self.ldpc_k, dtype=int)

        for i in range(n_blocks):
            block_llr = llr_for_decode[i * self.ldpc.n:(i + 1) * self.ldpc.n]
            codeword, success = self.ldpc.decode_minsum(block_llr, max_iter=20)
            decoded_bits[i * self.ldpc_k:(i + 1) * self.ldpc_k] = self.ldpc.get_info_bits(codeword)

        # Cắt về đúng số bit gốc
        decoded_bits = decoded_bits[:n_info_bits]

        return decoded_bits, detected_symbols

    def simulate_ber_ser(self, snr_range: np.ndarray,
                         n_bits_per_snr: int = 10000,
                         detection_method: str = 'mmse') -> dict:
        """
        Chạy mô phỏng BER và SER.

        Args:
            snr_range: Dải SNR cần mô phỏng
            n_bits_per_snr: Số bit cho mỗi điểm SNR
            detection_method: 'zf' hoặc 'mmse'

        Returns:
            Dictionary chứa kết quả BER và SER
        """
        ber_coded = []
        ber_uncoded = []
        ser = []

        print(f"\nRunning simulation with {detection_method.upper()} detection...")
        print(f"Bits per SNR point: {n_bits_per_snr}")
        print("-" * 50)

        for snr_db in tqdm(snr_range, desc="SNR points"):
            np.random.seed(self.seed)

            # Generate random bits
            info_bits = np.random.randint(0, 2, n_bits_per_snr)

            # Transmit
            tx_freq, tx_symbols, n_bits, n_blocks = self.transmit(info_bits)

            # Channel
            rx_freq, H_all, noise_var = self.channel_propagation(tx_freq, snr_db)

            # Receive
            decoded_bits, rx_symbols = self.receive(
                rx_freq, H_all, noise_var, n_bits, n_blocks, detection_method
            )

            # Calculate BER (coded)
            bit_errors = np.sum(info_bits != decoded_bits)
            current_ber = bit_errors / len(info_bits)
            ber_coded.append(current_ber)

            # Calculate SER (before decoding)
            # So sánh với constellation gần nhất
            tx_used = tx_symbols[:len(rx_symbols)]
            tx_indices = np.array([np.argmin(np.abs(s - self.qam.constellation)) for s in tx_used])
            rx_indices = np.array([np.argmin(np.abs(s - self.qam.constellation)) for s in rx_symbols])
            symbol_errors = np.sum(tx_indices != rx_indices)
            current_ser = symbol_errors / len(rx_symbols)
            ser.append(current_ser)

            # BER uncoded (hard decision without LDPC)
            rx_bits_hard = self.qam.demodulate_hard(rx_symbols)
            # Lấy bits tương ứng
            coded_bits_original = np.zeros(n_blocks * self.ldpc.n, dtype=int)
            for i in range(n_blocks):
                msg = np.zeros(self.ldpc_k, dtype=int)
                if i * self.ldpc_k < n_bits:
                    end_idx = min((i + 1) * self.ldpc_k, n_bits)
                    msg[:end_idx - i * self.ldpc_k] = info_bits[i * self.ldpc_k:end_idx]
                coded_bits_original[i * self.ldpc.n:(i + 1) * self.ldpc.n] = self.ldpc.encode(msg)

            # Compare uncoded
            rx_bits_needed = rx_bits_hard[:len(coded_bits_original)]
            uncoded_errors = np.sum(coded_bits_original != rx_bits_needed)
            ber_uncoded.append(uncoded_errors / len(coded_bits_original))

        return {
            'snr': snr_range,
            'ber_coded': np.array(ber_coded),
            'ber_uncoded': np.array(ber_uncoded),
            'ser': np.array(ser),
            'detection': detection_method
        }


def plot_results(results_list: list, save_path: str = None):
    """
    Vẽ đồ thị kết quả BER và SER.

    Args:
        results_list: Danh sách kết quả từ các mô phỏng
        save_path: Đường dẫn lưu hình (optional)
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    colors = ['blue', 'red', 'green', 'orange']
    markers = ['o', 's', '^', 'd']

    # BER plot
    ax1 = axes[0]
    for i, results in enumerate(results_list):
        snr = results['snr']
        detection = results['detection'].upper()

        # BER coded
        ber_coded = results['ber_coded']
        ber_coded[ber_coded == 0] = 1e-7  # Avoid log(0)
        ax1.semilogy(snr, ber_coded, f'{markers[i]}-',
                     color=colors[i], linewidth=2, markersize=8,
                     label=f'BER Coded ({detection})')

        # BER uncoded
        ber_uncoded = results['ber_uncoded']
        ber_uncoded[ber_uncoded == 0] = 1e-7
        ax1.semilogy(snr, ber_uncoded, f'{markers[i]}--',
                     color=colors[i], linewidth=1.5, markersize=6,
                     alpha=0.6, label=f'BER Uncoded ({detection})')

    ax1.set_xlabel('SNR (dB)', fontsize=12)
    ax1.set_ylabel('Bit Error Rate (BER)', fontsize=12)
    ax1.set_title('BER vs SNR: 2×2 MIMO-OFDM with LDPC & 64-QAM', fontsize=14)
    ax1.grid(True, which='both', alpha=0.3)
    ax1.legend(loc='lower left', fontsize=10)
    ax1.set_ylim([1e-6, 1])

    # SER plot
    ax2 = axes[1]
    for i, results in enumerate(results_list):
        snr = results['snr']
        detection = results['detection'].upper()

        ser = results['ser']
        ser[ser == 0] = 1e-7
        ax2.semilogy(snr, ser, f'{markers[i]}-',
                     color=colors[i], linewidth=2, markersize=8,
                     label=f'SER ({detection})')

    ax2.set_xlabel('SNR (dB)', fontsize=12)
    ax2.set_ylabel('Symbol Error Rate (SER)', fontsize=12)
    ax2.set_title('SER vs SNR: 2×2 MIMO-OFDM with 64-QAM', fontsize=14)
    ax2.grid(True, which='both', alpha=0.3)
    ax2.legend(loc='lower left', fontsize=10)
    ax2.set_ylim([1e-5, 1])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nFigure saved to: {save_path}")

    plt.show()


def print_summary(results: dict):
    """
    In bảng tóm tắt kết quả.
    """
    print("\n" + "=" * 70)
    print(f"SIMULATION RESULTS SUMMARY ({results['detection'].upper()} Detection)")
    print("=" * 70)
    print(f"{'SNR (dB)':<12}{'BER (Coded)':<18}{'BER (Uncoded)':<18}{'SER':<15}{'Coding Gain':<12}")
    print("-" * 70)

    for i in range(len(results['snr'])):
        snr = results['snr'][i]
        ber_c = results['ber_coded'][i]
        ber_u = results['ber_uncoded'][i]
        ser = results['ser'][i]

        if ber_u > 0 and ber_c > 0:
            gain = 10 * np.log10(ber_u / ber_c)
        else:
            gain = float('inf')

        print(f"{snr:<12.1f}{ber_c:<18.6e}{ber_u:<18.6e}{ser:<15.6e}{gain:<12.2f} dB")

    print("=" * 70)


def main():
    """
    Chương trình chính.
    """
    print("=" * 70)
    print("  2×2 MIMO-OFDM SYSTEM WITH LDPC CODING SIMULATION")
    print("  Rayleigh Fading Channel + AWGN | 64-QAM Modulation")
    print("=" * 70)

    # System parameters
    system = MIMOOFDMLDPCSystem(
        n_fft=64,
        n_cp=16,
        n_tx=2,
        n_rx=2,
        ldpc_n=96,      # Nhỏ hơn để chạy nhanh (production: 648 hoặc lớn hơn)
        ldpc_rate=0.5,
        seed=42
    )

    # SNR range
    snr_range = np.arange(0, 26, 2)

    # Run simulations
    results_mmse = system.simulate_ber_ser(
        snr_range=snr_range,
        n_bits_per_snr=5000,    # Tăng lên để kết quả chính xác hơn
        detection_method='mmse'
    )

    results_zf = system.simulate_ber_ser(
        snr_range=snr_range,
        n_bits_per_snr=5000,
        detection_method='zf'
    )

    # Print summaries
    print_summary(results_mmse)
    print_summary(results_zf)

    # Plot results
    plot_results([results_mmse, results_zf],
                 save_path='ber_ser_results.png')

    print("\n" + "=" * 70)
    print("SIMULATION COMPLETED!")
    print("=" * 70)

    return results_mmse, results_zf


if __name__ == "__main__":
    results = main()
