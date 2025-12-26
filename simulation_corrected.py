"""
2×2 MIMO-OFDM System with LDPC Coding - CORRECTED VERSION
==========================================================

Các sửa lỗi so với phiên bản gốc:
1. SNR definition: noise variance cố định trước kênh (Es/N0)
2. Post-detection noise variance cho soft demodulation
3. Thêm interleaving để phân tán burst errors
4. Block fading model (kênh không đổi trong 1 codeword)
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from qam_modulation import QAM64
from ldpc_codec import LDPCCode
from mimo_detector import MIMODetector
from interleaver import RandomInterleaver


class MIMOOFDMLDPCSystemCorrected:
    """
    Hệ thống 2×2 MIMO-OFDM với LDPC - Phiên bản đã sửa lỗi.
    """

    def __init__(self,
                 n_tx: int = 2,
                 n_rx: int = 2,
                 ldpc_n: int = 96,
                 ldpc_rate: float = 0.5,
                 seed: int = 42):
        """
        Khởi tạo hệ thống.
        """
        self.n_tx = n_tx
        self.n_rx = n_rx
        self.seed = seed

        print("Initializing CORRECTED system...")

        # 64-QAM modulator
        self.qam = QAM64()
        self.bits_per_symbol = 6

        # LDPC codec
        self.ldpc = LDPCCode(n=ldpc_n, rate=ldpc_rate, seed=seed)
        self.ldpc_k = self.ldpc.k
        self.ldpc_n = self.ldpc.n

        # Interleaver (sau LDPC, trước QAM)
        self.interleaver = RandomInterleaver(length=self.ldpc_n, seed=seed + 1)

        # MIMO detector
        self.detector = MIMODetector(n_tx=n_tx, n_rx=n_rx)

        print(f"System config:")
        print(f"  - MIMO: {n_tx}×{n_rx}")
        print(f"  - LDPC: ({self.ldpc_n}, {self.ldpc_k}), rate = {ldpc_rate}")
        print(f"  - Modulation: 64-QAM")
        print(f"  - Interleaver: Random, length = {self.ldpc_n}")

    def simulate_one_block(self, snr_db: float, use_interleaver: bool = True) -> dict:
        """
        Mô phỏng một block LDPC.

        Args:
            snr_db: SNR in dB (Es/N0)
            use_interleaver: Có sử dụng interleaver không

        Returns:
            Dictionary với BER và các thông số khác
        """
        # 1. Generate random message
        msg = np.random.randint(0, 2, self.ldpc_k)

        # 2. LDPC Encode
        codeword = self.ldpc.encode(msg)

        # 3. Interleave (optional)
        if use_interleaver:
            codeword_int = self.interleaver.interleave(codeword)
        else:
            codeword_int = codeword

        # 4. Pad for 64-QAM (6 bits/symbol)
        n_bits = len(codeword_int)
        n_pad = (self.bits_per_symbol - n_bits % self.bits_per_symbol) % self.bits_per_symbol
        bits_padded = np.concatenate([codeword_int, np.zeros(n_pad, dtype=int)])

        # 5. 64-QAM Modulation
        symbols = self.qam.modulate(bits_padded)

        # 6. Chia cho MIMO (2 streams)
        n_symbols = len(symbols)
        n_mimo_uses = n_symbols // self.n_tx
        if n_symbols % self.n_tx != 0:
            # Pad thêm
            n_mimo_uses += 1
            symbols = np.concatenate([symbols, np.zeros(n_mimo_uses * self.n_tx - n_symbols, dtype=complex)])

        # 7. MIMO Transmission với Block Fading
        # Kênh KHÔNG ĐỔI trong toàn bộ codeword (block fading)
        H = (np.random.randn(self.n_rx, self.n_tx) +
             1j * np.random.randn(self.n_rx, self.n_tx)) / np.sqrt(2)

        # SNR definition: Es/N0 với Es = E[|x|²] = 1
        snr_linear = 10 ** (snr_db / 10)
        noise_var = 1.0 / snr_linear  # Cố định, không phụ thuộc kênh!

        detected_symbols = []
        post_noise_vars = []

        for i in range(n_mimo_uses):
            x = symbols[i * self.n_tx:(i + 1) * self.n_tx]

            # Channel: y = H @ x + n
            y_clean = H @ x
            noise = np.sqrt(noise_var / 2) * (
                np.random.randn(self.n_rx) + 1j * np.random.randn(self.n_rx)
            )
            y = y_clean + noise

            # MMSE Detection with post-detection variance
            x_hat, post_var = self.detector.detect_mmse(y, H, noise_var, return_post_var=True)

            detected_symbols.extend(x_hat)
            post_noise_vars.extend(post_var)

        detected_symbols = np.array(detected_symbols[:n_symbols])
        post_noise_vars = np.array(post_noise_vars[:n_symbols])

        # 8. Soft Demodulation với CORRECT noise variance
        # Mỗi symbol có noise variance khác nhau (theo stream)
        llr_all = []
        for i, (sym, var) in enumerate(zip(detected_symbols, post_noise_vars)):
            # Soft demod với variance đúng
            llr_sym = self.qam.demodulate_soft(np.array([sym]), var)
            llr_all.extend(llr_sym)

        llr = np.array(llr_all)

        # Bỏ padding
        llr = llr[:n_bits]

        # 9. Deinterleave
        if use_interleaver:
            llr = self.interleaver.deinterleave(llr)

        # 10. LDPC Decode
        decoded_codeword, success = self.ldpc.decode_minsum(llr, max_iter=30)
        decoded_msg = self.ldpc.get_info_bits(decoded_codeword)

        # 11. Calculate BER
        ber_coded = np.mean(msg != decoded_msg)

        # Uncoded BER (hard decision on coded bits)
        rx_bits_hard = (llr < 0).astype(int)
        if use_interleaver:
            # So sánh với codeword gốc (không interleaved)
            ber_uncoded = np.mean(codeword != rx_bits_hard)
        else:
            ber_uncoded = np.mean(codeword != rx_bits_hard)

        return {
            'ber_coded': ber_coded,
            'ber_uncoded': ber_uncoded,
            'ldpc_success': success
        }

    def simulate_ber(self, snr_range: np.ndarray, n_blocks: int = 100,
                     use_interleaver: bool = True) -> dict:
        """
        Chạy mô phỏng BER qua dải SNR.

        Args:
            snr_range: Dải SNR (dB)
            n_blocks: Số block LDPC mỗi điểm SNR
            use_interleaver: Có dùng interleaver không

        Returns:
            Dictionary kết quả
        """
        ber_coded_list = []
        ber_uncoded_list = []

        print(f"\nSimulating with interleaver={'ON' if use_interleaver else 'OFF'}...")

        for snr_db in tqdm(snr_range, desc="SNR points"):
            np.random.seed(self.seed)

            ber_coded_sum = 0
            ber_uncoded_sum = 0

            for _ in range(n_blocks):
                result = self.simulate_one_block(snr_db, use_interleaver)
                ber_coded_sum += result['ber_coded']
                ber_uncoded_sum += result['ber_uncoded']

            ber_coded_list.append(ber_coded_sum / n_blocks)
            ber_uncoded_list.append(ber_uncoded_sum / n_blocks)

        return {
            'snr': snr_range,
            'ber_coded': np.array(ber_coded_list),
            'ber_uncoded': np.array(ber_uncoded_list),
            'interleaver': use_interleaver
        }


def plot_comparison(results_with_int, results_without_int, save_path=None):
    """
    So sánh kết quả với và không có interleaver.
    """
    fig, ax = plt.subplots(figsize=(10, 7))

    snr = results_with_int['snr']

    # Với interleaver
    ber_coded_int = results_with_int['ber_coded'].copy()
    ber_uncoded_int = results_with_int['ber_uncoded'].copy()
    ber_coded_int[ber_coded_int == 0] = 1e-7
    ber_uncoded_int[ber_uncoded_int == 0] = 1e-7

    ax.semilogy(snr, ber_coded_int, 'bo-', linewidth=2, markersize=8,
                label='BER Coded (with interleaver)')
    ax.semilogy(snr, ber_uncoded_int, 'b^--', linewidth=1.5, markersize=6,
                alpha=0.7, label='BER Uncoded (with interleaver)')

    # Không có interleaver
    ber_coded_no = results_without_int['ber_coded'].copy()
    ber_uncoded_no = results_without_int['ber_uncoded'].copy()
    ber_coded_no[ber_coded_no == 0] = 1e-7
    ber_uncoded_no[ber_uncoded_no == 0] = 1e-7

    ax.semilogy(snr, ber_coded_no, 'ro-', linewidth=2, markersize=8,
                label='BER Coded (no interleaver)')
    ax.semilogy(snr, ber_uncoded_no, 'r^--', linewidth=1.5, markersize=6,
                alpha=0.7, label='BER Uncoded (no interleaver)')

    ax.set_xlabel('SNR (dB) - Es/N0', fontsize=12)
    ax.set_ylabel('Bit Error Rate (BER)', fontsize=12)
    ax.set_title('2×2 MIMO + LDPC + 64-QAM over Rayleigh Channel\n(Block Fading, Corrected SNR)', fontsize=14)
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(loc='lower left', fontsize=10)
    ax.set_ylim([1e-5, 1])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Figure saved to: {save_path}")

    plt.show()


def print_results(results: dict, title: str):
    """In bảng kết quả."""
    print(f"\n{'='*60}")
    print(f"{title}")
    print(f"{'='*60}")
    print(f"{'SNR (dB)':<12}{'BER Coded':<18}{'BER Uncoded':<18}{'Coding Gain':<12}")
    print("-" * 60)

    for i in range(len(results['snr'])):
        snr = results['snr'][i]
        ber_c = results['ber_coded'][i]
        ber_u = results['ber_uncoded'][i]

        if ber_u > 0 and ber_c > 0 and ber_c < ber_u:
            gain = 10 * np.log10(ber_u / ber_c)
            gain_str = f"+{gain:.2f} dB"
        elif ber_c == 0:
            gain_str = "∞"
        else:
            gain_str = "N/A"

        print(f"{snr:<12.1f}{ber_c:<18.6e}{ber_u:<18.6e}{gain_str:<12}")

    print("=" * 60)


def main():
    """Chương trình chính."""
    print("=" * 70)
    print("  2×2 MIMO-OFDM + LDPC + 64-QAM SIMULATION (CORRECTED)")
    print("  Block Fading Rayleigh Channel | Correct SNR Definition")
    print("=" * 70)

    system = MIMOOFDMLDPCSystemCorrected(
        n_tx=2,
        n_rx=2,
        ldpc_n=96,
        ldpc_rate=0.5,
        seed=42
    )

    # SNR range cao hơn cho 64-QAM
    snr_range = np.arange(10, 32, 2)

    # Chạy với interleaver
    results_with = system.simulate_ber(snr_range, n_blocks=50, use_interleaver=True)

    # Chạy không có interleaver
    results_without = system.simulate_ber(snr_range, n_blocks=50, use_interleaver=False)

    # In kết quả
    print_results(results_with, "RESULTS WITH INTERLEAVER")
    print_results(results_without, "RESULTS WITHOUT INTERLEAVER")

    # Vẽ đồ thị
    plot_comparison(results_with, results_without, save_path='ber_corrected.png')

    print("\n" + "=" * 70)
    print("SIMULATION COMPLETED!")
    print("=" * 70)

    return results_with, results_without


if __name__ == "__main__":
    results = main()
