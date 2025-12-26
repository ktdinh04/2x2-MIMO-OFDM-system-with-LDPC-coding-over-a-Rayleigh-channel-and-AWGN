"""
Module: 64-QAM Modulation/Demodulation
=======================================
Bộ điều chế và giải điều chế 64-QAM với ánh xạ Gray.

64-QAM sử dụng 6 bit/symbol với constellation 8x8.
- 3 bit cho phần thực (I)
- 3 bit cho phần ảo (Q)

Công suất trung bình được chuẩn hóa về 1.
"""

import numpy as np
from typing import Tuple


class QAM64:
    """
    Bộ điều chế/giải điều chế 64-QAM.

    Attributes:
        M (int): Số điểm constellation (64)
        bits_per_symbol (int): Số bit trên mỗi symbol (6)
        constellation (np.ndarray): Bảng constellation 64 điểm
        gray_map (np.ndarray): Ánh xạ Gray code
        normalization_factor (float): Hệ số chuẩn hóa công suất
    """

    def __init__(self):
        self.M = 64
        self.bits_per_symbol = 6
        self.constellation, self.gray_map = self._create_constellation()
        self.normalization_factor = self._calculate_normalization()
        # Chuẩn hóa constellation về công suất đơn vị
        self.constellation = self.constellation / self.normalization_factor

    def _gray_code(self, n: int) -> list:
        """
        Tạo Gray code cho n bit.

        Gray code đảm bảo các symbol lân cận chỉ khác nhau 1 bit,
        giúp giảm BER khi có lỗi symbol.

        Args:
            n: Số bit

        Returns:
            List các giá trị Gray code
        """
        if n == 0:
            return [0]
        if n == 1:
            return [0, 1]

        # Đệ quy tạo Gray code
        prev = self._gray_code(n - 1)
        result = []
        for i in prev:
            result.append(i)
        for i in reversed(prev):
            result.append(i + (1 << (n - 1)))
        return result

    def _create_constellation(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Tạo constellation 64-QAM với ánh xạ Gray.

        Constellation được sắp xếp theo lưới 8x8 với khoảng cách 2
        giữa các điểm lân cận.

        Returns:
            constellation: Mảng 64 điểm phức
            gray_map: Mảng ánh xạ từ index sang Gray code
        """
        # Gray code cho 3 bit (I và Q riêng biệt)
        gray_3bit = self._gray_code(3)  # [0, 1, 3, 2, 6, 7, 5, 4]

        constellation = np.zeros(64, dtype=complex)
        gray_map = np.zeros(64, dtype=int)

        # Các mức PAM-8: -7, -5, -3, -1, +1, +3, +5, +7
        levels = np.array([-7, -5, -3, -1, 1, 3, 5, 7])

        for i in range(8):  # Phần thực
            for j in range(8):  # Phần ảo
                # Tính index dựa trên Gray code
                gray_i = gray_3bit[i]
                gray_j = gray_3bit[j]

                # 6-bit symbol: [I2 I1 I0 Q2 Q1 Q0]
                symbol_index = (gray_i << 3) | gray_j

                # Điểm constellation
                real_part = levels[i]
                imag_part = levels[j]

                constellation[symbol_index] = real_part + 1j * imag_part
                gray_map[symbol_index] = symbol_index

        return constellation, gray_map

    def _calculate_normalization(self) -> float:
        """
        Tính hệ số chuẩn hóa để công suất trung bình = 1.

        E[|s|^2] = 1 sau chuẩn hóa.

        Returns:
            Hệ số chuẩn hóa (căn bậc 2 của công suất trung bình)
        """
        avg_power = np.mean(np.abs(self.constellation) ** 2)
        return np.sqrt(avg_power)

    def modulate(self, bits: np.ndarray) -> np.ndarray:
        """
        Điều chế chuỗi bit thành symbols 64-QAM.

        Args:
            bits: Mảng bit đầu vào (chiều dài phải chia hết cho 6)

        Returns:
            Mảng các symbol phức đã điều chế

        Raises:
            ValueError: Nếu số bit không chia hết cho 6
        """
        if len(bits) % self.bits_per_symbol != 0:
            raise ValueError(f"Số bit phải chia hết cho {self.bits_per_symbol}")

        num_symbols = len(bits) // self.bits_per_symbol
        symbols = np.zeros(num_symbols, dtype=complex)

        for i in range(num_symbols):
            # Lấy 6 bit cho mỗi symbol
            bit_group = bits[i * 6:(i + 1) * 6]

            # Chuyển đổi 6 bit sang index thập phân
            index = 0
            for bit in bit_group:
                index = (index << 1) | int(bit)

            symbols[i] = self.constellation[index]

        return symbols

    def demodulate_hard(self, received: np.ndarray) -> np.ndarray:
        """
        Giải điều chế hard-decision (quyết định cứng).

        Tìm điểm constellation gần nhất với tín hiệu nhận được.

        Args:
            received: Mảng symbol phức nhận được

        Returns:
            Mảng bit đã giải điều chế
        """
        num_symbols = len(received)
        bits = np.zeros(num_symbols * self.bits_per_symbol, dtype=int)

        for i in range(num_symbols):
            # Tìm điểm constellation gần nhất
            distances = np.abs(received[i] - self.constellation)
            index = np.argmin(distances)

            # Chuyển index sang bit
            for j in range(self.bits_per_symbol - 1, -1, -1):
                bits[i * 6 + (5 - j)] = (index >> j) & 1

        return bits

    def demodulate_soft(self, received: np.ndarray, noise_var: float) -> np.ndarray:
        """
        Giải điều chế soft-decision (quyết định mềm) - tính LLR.

        Log-Likelihood Ratio (LLR) cho mỗi bit:
        LLR(b_k) = log(P(b_k=0|r) / P(b_k=1|r))

        Giá trị dương → nghiêng về bit 0
        Giá trị âm → nghiêng về bit 1

        Args:
            received: Mảng symbol phức nhận được
            noise_var: Phương sai nhiễu (sigma^2)

        Returns:
            Mảng LLR cho từng bit
        """
        num_symbols = len(received)
        llr = np.zeros(num_symbols * self.bits_per_symbol)

        for i in range(num_symbols):
            r = received[i]

            for k in range(self.bits_per_symbol):
                # Tìm các điểm constellation có bit k = 0 và = 1
                bit_position = self.bits_per_symbol - 1 - k

                # Symbols có bit k = 0
                idx_0 = [j for j in range(self.M) if not ((j >> bit_position) & 1)]
                # Symbols có bit k = 1
                idx_1 = [j for j in range(self.M) if (j >> bit_position) & 1]

                # Tính khoảng cách Euclidean bình phương
                dist_0 = np.abs(r - self.constellation[idx_0]) ** 2
                dist_1 = np.abs(r - self.constellation[idx_1]) ** 2

                # Max-log approximation cho LLR
                # LLR ≈ (min distance for bit=1 - min distance for bit=0) / noise_var
                min_dist_0 = np.min(dist_0)
                min_dist_1 = np.min(dist_1)

                llr[i * 6 + k] = (min_dist_1 - min_dist_0) / noise_var

        return llr

    def get_constellation_points(self) -> np.ndarray:
        """
        Trả về các điểm constellation.

        Returns:
            Mảng 64 điểm phức của constellation
        """
        return self.constellation.copy()

    def calculate_ser(self, tx_symbols: np.ndarray, rx_symbols: np.ndarray) -> float:
        """
        Tính Symbol Error Rate (SER).

        Args:
            tx_symbols: Symbols đã truyền
            rx_symbols: Symbols nhận được (sau giải điều chế hard)

        Returns:
            Tỷ lệ lỗi symbol
        """
        # Giải điều chế cả hai để so sánh index
        tx_indices = np.zeros(len(tx_symbols), dtype=int)
        rx_indices = np.zeros(len(rx_symbols), dtype=int)

        for i in range(len(tx_symbols)):
            tx_indices[i] = np.argmin(np.abs(tx_symbols[i] - self.constellation))
            rx_indices[i] = np.argmin(np.abs(rx_symbols[i] - self.constellation))

        errors = np.sum(tx_indices != rx_indices)
        return errors / len(tx_symbols)


def plot_constellation(qam: QAM64, received: np.ndarray = None):
    """
    Vẽ constellation diagram.

    Args:
        qam: Đối tượng QAM64
        received: Tín hiệu nhận được (tùy chọn)
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 10))

    # Vẽ constellation gốc
    constellation = qam.get_constellation_points()
    ax.scatter(constellation.real, constellation.imag,
               c='blue', s=100, marker='o', label='64-QAM Constellation')

    # Đánh số các điểm
    for i, point in enumerate(constellation):
        ax.annotate(f'{i:06b}', (point.real, point.imag),
                   textcoords="offset points", xytext=(0, 10),
                   ha='center', fontsize=6)

    # Vẽ tín hiệu nhận được nếu có
    if received is not None:
        ax.scatter(received.real, received.imag,
                  c='red', s=20, marker='x', alpha=0.5, label='Received')

    ax.set_xlabel('In-phase (I)')
    ax.set_ylabel('Quadrature (Q)')
    ax.set_title('64-QAM Constellation Diagram')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.axis('equal')

    return fig


# Test module
if __name__ == "__main__":
    print("=" * 60)
    print("TEST MODULE: 64-QAM Modulation/Demodulation")
    print("=" * 60)

    qam = QAM64()

    # Test với dữ liệu ngẫu nhiên
    np.random.seed(42)
    num_bits = 600  # 100 symbols
    tx_bits = np.random.randint(0, 2, num_bits)

    # Điều chế
    tx_symbols = qam.modulate(tx_bits)
    print(f"\nSố bit đầu vào: {num_bits}")
    print(f"Số symbols: {len(tx_symbols)}")
    print(f"Công suất trung bình: {np.mean(np.abs(tx_symbols)**2):.4f}")

    # Thêm nhiễu AWGN
    snr_db = 20
    snr_linear = 10 ** (snr_db / 10)
    noise_var = 1 / snr_linear
    noise = np.sqrt(noise_var / 2) * (np.random.randn(len(tx_symbols)) +
                                       1j * np.random.randn(len(tx_symbols)))
    rx_symbols = tx_symbols + noise

    # Giải điều chế hard
    rx_bits_hard = qam.demodulate_hard(rx_symbols)
    ber_hard = np.mean(tx_bits != rx_bits_hard)
    print(f"\nSNR: {snr_db} dB")
    print(f"BER (hard decision): {ber_hard:.6f}")

    # Giải điều chế soft
    llr = qam.demodulate_soft(rx_symbols, noise_var)
    rx_bits_soft = (llr < 0).astype(int)
    ber_soft = np.mean(tx_bits != rx_bits_soft)
    print(f"BER (soft decision): {ber_soft:.6f}")

    print("\n" + "=" * 60)
    print("Test PASSED!")
    print("=" * 60)
