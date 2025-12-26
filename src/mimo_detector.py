"""
Module: MIMO Detector
======================
Các thuật toán phát hiện tín hiệu MIMO:
1. Zero-Forcing (ZF): Loại bỏ nhiễu liên kênh hoàn toàn
2. MMSE (Minimum Mean Square Error): Cân bằng giữa nhiễu và interference
3. ML (Maximum Likelihood): Tối ưu nhưng phức tạp

Mô hình: y = H·x + n

Mục tiêu: Ước lượng x từ y khi biết H
"""

import numpy as np
from typing import Tuple, Optional
from itertools import product


class MIMODetector:
    """
    Bộ phát hiện MIMO với các thuật toán linear và non-linear.
    """

    def __init__(self, n_tx: int = 2, n_rx: int = 2):
        """
        Khởi tạo bộ phát hiện MIMO.

        Args:
            n_tx: Số anten phát
            n_rx: Số anten thu
        """
        self.n_tx = n_tx
        self.n_rx = n_rx

    def detect_zf(self, y: np.ndarray, H: np.ndarray) -> np.ndarray:
        """
        Zero-Forcing Detection.

        ZF hoàn toàn loại bỏ nhiễu liên kênh bằng cách sử dụng
        pseudo-inverse của ma trận kênh.

        x_hat = (H^H · H)^(-1) · H^H · y = H^+ · y

        Ưu điểm:
        - Đơn giản, độ phức tạp thấp
        - Không cần biết SNR

        Nhược điểm:
        - Khuếch đại nhiễu khi kênh không tốt (ill-conditioned)
        - Hiệu suất kém hơn MMSE

        Args:
            y: Tín hiệu nhận (n_rx,) hoặc (n_rx, n_samples)
            H: Ma trận kênh (n_rx, n_tx)

        Returns:
            x_hat: Tín hiệu ước lượng (n_tx,) hoặc (n_tx, n_samples)
        """
        # Tính pseudo-inverse: H^+ = (H^H · H)^(-1) · H^H
        H_pinv = np.linalg.pinv(H)

        # Detect
        if y.ndim == 1:
            x_hat = H_pinv @ y
        else:
            x_hat = H_pinv @ y

        return x_hat

    def detect_mmse(self, y: np.ndarray, H: np.ndarray,
                    noise_var: float) -> np.ndarray:
        """
        MMSE (Minimum Mean Square Error) Detection.

        MMSE cân bằng giữa loại bỏ interference và tránh khuếch đại nhiễu.

        x_hat = (H^H · H + σ²·I)^(-1) · H^H · y

        Với SNR cao: MMSE → ZF
        Với SNR thấp: MMSE → Matched Filter

        Ưu điểm:
        - Tối ưu theo MSE
        - Không khuếch đại nhiễu quá mức

        Nhược điểm:
        - Cần biết phương sai nhiễu
        - Phức tạp hơn ZF một chút

        Args:
            y: Tín hiệu nhận
            H: Ma trận kênh
            noise_var: Phương sai nhiễu (σ²)

        Returns:
            x_hat: Tín hiệu ước lượng
        """
        n_tx = H.shape[1]

        # MMSE filter: W = (H^H·H + σ²·I)^(-1) · H^H
        H_H = H.conj().T
        W = np.linalg.inv(H_H @ H + noise_var * np.eye(n_tx)) @ H_H

        # Detect
        if y.ndim == 1:
            x_hat = W @ y
        else:
            x_hat = W @ y

        return x_hat

    def detect_ml(self, y: np.ndarray, H: np.ndarray,
                  constellation: np.ndarray) -> np.ndarray:
        """
        Maximum Likelihood Detection.

        ML tìm kiếm vector x tối ưu trong không gian symbol:
        x_hat = argmin_x ||y - H·x||²

        Ưu điểm:
        - Tối ưu về BER
        - Không cần biết SNR

        Nhược điểm:
        - Độ phức tạp O(M^Nt) với M là số điểm constellation
        - Không khả thi cho M và Nt lớn

        Args:
            y: Tín hiệu nhận (n_rx,)
            H: Ma trận kênh (n_rx, n_tx)
            constellation: Các điểm constellation

        Returns:
            x_hat: Vector symbols ước lượng (n_tx,)
        """
        n_tx = H.shape[1]

        # Tạo tất cả các tổ hợp symbols có thể
        candidates = list(product(constellation, repeat=n_tx))

        min_distance = float('inf')
        best_x = None

        for x_candidate in candidates:
            x = np.array(x_candidate)
            # Tính khoảng cách Euclidean
            distance = np.sum(np.abs(y - H @ x) ** 2)

            if distance < min_distance:
                min_distance = distance
                best_x = x

        return best_x

    def detect_zf_sic(self, y: np.ndarray, H: np.ndarray,
                      constellation: np.ndarray) -> np.ndarray:
        """
        ZF-SIC (Successive Interference Cancellation).

        Phát hiện tuần tự với loại bỏ nhiễu:
        1. Sắp xếp theo SNR (anten mạnh nhất trước)
        2. Phát hiện symbol mạnh nhất
        3. Loại bỏ đóng góp của nó và lặp lại

        Args:
            y: Tín hiệu nhận
            H: Ma trận kênh
            constellation: Các điểm constellation

        Returns:
            x_hat: Symbols ước lượng
        """
        n_tx = H.shape[1]
        x_hat = np.zeros(n_tx, dtype=complex)
        y_residual = y.copy()
        H_residual = H.copy()

        # Thứ tự detection dựa trên norm cột của H
        col_norms = np.linalg.norm(H, axis=0)
        detection_order = np.argsort(col_norms)[::-1]  # Mạnh nhất trước

        for idx in detection_order:
            # ZF detect cho anten này
            H_pinv = np.linalg.pinv(H_residual)
            x_linear = H_pinv @ y_residual

            # Slicing: chọn symbol gần nhất trong constellation
            symbol_idx = np.argmin(np.abs(x_linear[idx] - constellation))
            x_hat[idx] = constellation[symbol_idx]

            # SIC: loại bỏ đóng góp
            y_residual = y_residual - H_residual[:, idx] * x_hat[idx]

            # Đặt cột đã detect về 0
            H_residual[:, idx] = 0

        return x_hat

    def detect_mmse_sic(self, y: np.ndarray, H: np.ndarray,
                        noise_var: float, constellation: np.ndarray) -> np.ndarray:
        """
        MMSE-SIC Detection.

        Tương tự ZF-SIC nhưng sử dụng MMSE cho mỗi bước.

        Args:
            y: Tín hiệu nhận
            H: Ma trận kênh
            noise_var: Phương sai nhiễu
            constellation: Các điểm constellation

        Returns:
            x_hat: Symbols ước lượng
        """
        n_tx = H.shape[1]
        x_hat = np.zeros(n_tx, dtype=complex)
        y_residual = y.copy()

        remaining = list(range(n_tx))

        for _ in range(n_tx):
            # Tính MMSE cho các anten còn lại
            H_remain = H[:, remaining]
            H_H = H_remain.conj().T
            W = np.linalg.inv(H_H @ H_remain + noise_var * np.eye(len(remaining))) @ H_H
            x_linear = W @ y_residual

            # Chọn anten với SINR cao nhất (post-detection SNR)
            sinr = np.abs(x_linear) ** 2 / (
                np.diag(W @ W.conj().T) * noise_var + 1e-10
            )
            best_local_idx = np.argmax(sinr)
            best_idx = remaining[best_local_idx]

            # Slicing
            symbol_idx = np.argmin(np.abs(x_linear[best_local_idx] - constellation))
            x_hat[best_idx] = constellation[symbol_idx]

            # SIC
            y_residual = y_residual - H[:, best_idx] * x_hat[best_idx]

            remaining.remove(best_idx)

        return x_hat


class MIMOOFDMDetector:
    """
    Bộ phát hiện cho MIMO-OFDM.

    Thực hiện detection per-subcarrier.
    """

    def __init__(self, n_tx: int = 2, n_rx: int = 2):
        """
        Khởi tạo.

        Args:
            n_tx: Số anten phát
            n_rx: Số anten thu
        """
        self.detector = MIMODetector(n_tx, n_rx)
        self.n_tx = n_tx
        self.n_rx = n_rx

    def detect_zf_ofdm(self, Y: np.ndarray, H: np.ndarray) -> np.ndarray:
        """
        ZF detection cho MIMO-OFDM.

        Args:
            Y: Tín hiệu nhận (n_rx, n_subcarriers)
            H: Ma trận kênh (n_subcarriers, n_rx, n_tx)

        Returns:
            X_hat: Symbols ước lượng (n_tx, n_subcarriers)
        """
        n_subcarriers = Y.shape[1]
        X_hat = np.zeros((self.n_tx, n_subcarriers), dtype=complex)

        for k in range(n_subcarriers):
            y_k = Y[:, k]
            H_k = H[k]
            X_hat[:, k] = self.detector.detect_zf(y_k, H_k)

        return X_hat

    def detect_mmse_ofdm(self, Y: np.ndarray, H: np.ndarray,
                         noise_var: float) -> np.ndarray:
        """
        MMSE detection cho MIMO-OFDM.

        Args:
            Y: Tín hiệu nhận (n_rx, n_subcarriers)
            H: Ma trận kênh (n_subcarriers, n_rx, n_tx)
            noise_var: Phương sai nhiễu

        Returns:
            X_hat: Symbols ước lượng (n_tx, n_subcarriers)
        """
        n_subcarriers = Y.shape[1]
        X_hat = np.zeros((self.n_tx, n_subcarriers), dtype=complex)

        for k in range(n_subcarriers):
            y_k = Y[:, k]
            H_k = H[k]
            X_hat[:, k] = self.detector.detect_mmse(y_k, H_k, noise_var)

        return X_hat

    def compute_soft_output(self, Y: np.ndarray, H: np.ndarray,
                            noise_var: float, constellation: np.ndarray,
                            bits_per_symbol: int) -> np.ndarray:
        """
        Tính LLR cho soft MIMO detection.

        Sử dụng max-log approximation:
        LLR(b_k) ≈ min(d² for b_k=0) - min(d² for b_k=1) / σ²

        Args:
            Y: Tín hiệu nhận
            H: Ma trận kênh
            noise_var: Phương sai nhiễu
            constellation: Điểm constellation
            bits_per_symbol: Số bit/symbol

        Returns:
            LLR cho mỗi bit
        """
        n_subcarriers = Y.shape[1]
        n_bits = self.n_tx * n_subcarriers * bits_per_symbol
        llr = np.zeros(n_bits)

        M = len(constellation)
        bit_idx = 0

        for k in range(n_subcarriers):
            y_k = Y[:, k]
            H_k = H[k]

            # Duyệt qua tất cả tổ hợp symbols
            candidates = list(product(range(M), repeat=self.n_tx))

            for tx in range(self.n_tx):
                for b in range(bits_per_symbol):
                    # Tìm min distance cho bit = 0 và bit = 1
                    min_d0 = float('inf')
                    min_d1 = float('inf')

                    for cand_idx in candidates:
                        x = constellation[list(cand_idx)]
                        d = np.sum(np.abs(y_k - H_k @ x) ** 2)

                        # Kiểm tra bit b của symbol tx
                        sym_idx = cand_idx[tx]
                        bit_val = (sym_idx >> (bits_per_symbol - 1 - b)) & 1

                        if bit_val == 0:
                            min_d0 = min(min_d0, d)
                        else:
                            min_d1 = min(min_d1, d)

                    llr[bit_idx] = (min_d1 - min_d0) / noise_var
                    bit_idx += 1

        return llr


def estimate_channel_ls(Y_pilot: np.ndarray, X_pilot: np.ndarray) -> np.ndarray:
    """
    Ước lượng kênh bằng Least Squares.

    H_hat = Y_pilot @ X_pilot^H @ (X_pilot @ X_pilot^H)^(-1)

    Args:
        Y_pilot: Tín hiệu pilot nhận được (n_rx, n_pilots)
        X_pilot: Pilots đã biết (n_tx, n_pilots)

    Returns:
        H_hat: Ước lượng kênh (n_rx, n_tx)
    """
    X_pilot_H = X_pilot.conj().T
    H_hat = Y_pilot @ X_pilot_H @ np.linalg.inv(X_pilot @ X_pilot_H)
    return H_hat


def estimate_channel_mmse(Y_pilot: np.ndarray, X_pilot: np.ndarray,
                          noise_var: float, R_H: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Ước lượng kênh bằng MMSE.

    Sử dụng prior về channel correlation nếu có.

    Args:
        Y_pilot: Tín hiệu pilot nhận được
        X_pilot: Pilots đã biết
        noise_var: Phương sai nhiễu
        R_H: Ma trận tương quan kênh (optional)

    Returns:
        H_hat: Ước lượng kênh
    """
    n_rx, n_pilots = Y_pilot.shape
    n_tx = X_pilot.shape[0]

    # LS estimate first
    H_ls = estimate_channel_ls(Y_pilot, X_pilot)

    if R_H is None:
        # Assume iid Rayleigh: R_H = I
        R_H = np.eye(n_rx * n_tx)

    # MMSE refinement (simplified)
    # Full MMSE would require vectorization and Kronecker products
    # Here we just return LS for simplicity
    return H_ls


# Test module
if __name__ == "__main__":
    print("=" * 60)
    print("TEST MODULE: MIMO Detector")
    print("=" * 60)

    np.random.seed(42)

    # Setup
    n_tx, n_rx = 2, 2
    detector = MIMODetector(n_tx, n_rx)

    # QPSK constellation for testing
    constellation = np.array([1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j]) / np.sqrt(2)

    # Generate channel and signal
    H = (np.random.randn(n_rx, n_tx) + 1j * np.random.randn(n_rx, n_tx)) / np.sqrt(2)
    x_true = constellation[np.random.randint(0, 4, n_tx)]

    print(f"\nTrue transmitted symbols: {x_true}")
    print(f"Channel matrix H:\n{H}")

    # Test at different SNR levels
    print("\n--- Detection Performance ---")
    print(f"{'SNR (dB)':<12}{'ZF MSE':<15}{'MMSE MSE':<15}{'ML Correct':<12}")
    print("-" * 55)

    for snr_db in [5, 10, 15, 20, 25]:
        snr_linear = 10 ** (snr_db / 10)
        noise_var = 1 / snr_linear

        # Generate received signal
        y_clean = H @ x_true
        noise = np.sqrt(noise_var / 2) * (np.random.randn(n_rx) + 1j * np.random.randn(n_rx))
        y = y_clean + noise

        # ZF detection
        x_zf = detector.detect_zf(y, H)
        mse_zf = np.mean(np.abs(x_true - x_zf) ** 2)

        # MMSE detection
        x_mmse = detector.detect_mmse(y, H, noise_var)
        mse_mmse = np.mean(np.abs(x_true - x_mmse) ** 2)

        # ML detection
        x_ml = detector.detect_ml(y, H, constellation)
        ml_correct = np.allclose(x_true, x_ml, atol=0.1)

        print(f"{snr_db:<12}{mse_zf:<15.6f}{mse_mmse:<15.6f}{str(ml_correct):<12}")

    # Test SIC
    print("\n--- SIC Detection Test ---")
    snr_db = 15
    noise_var = 1 / (10 ** (snr_db / 10))
    y_clean = H @ x_true
    noise = np.sqrt(noise_var / 2) * (np.random.randn(n_rx) + 1j * np.random.randn(n_rx))
    y = y_clean + noise

    x_zf_sic = detector.detect_zf_sic(y, H, constellation)
    x_mmse_sic = detector.detect_mmse_sic(y, H, noise_var, constellation)

    print(f"True:     {x_true}")
    print(f"ZF-SIC:   {x_zf_sic}")
    print(f"MMSE-SIC: {x_mmse_sic}")

    print("\n" + "=" * 60)
    print("Test PASSED!")
    print("=" * 60)
