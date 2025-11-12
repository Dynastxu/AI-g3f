# EMD_Decomposer.py
import numpy as np
from PyEMD import EMD, EEMD, CEEMDAN
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq


def _get_dominant_frequency(signal, sampling_rate=1):
    """获取信号的主频率"""
    n = len(signal)
    yf = fft(signal)
    xf = fftfreq(n, 1 / sampling_rate)[:n // 2]

    # 找到幅度最大的频率
    idx = np.argmax(np.abs(yf[:n // 2]))
    dominant_freq = xf[idx]

    return dominant_freq


class SignalDecomposer:
    """信号分解器 - 支持EMD/EEMD/CEEMDAN"""

    def __init__(self):
        self.emd = EMD()
        self.eemd = EEMD()
        self.ceemdan = CEEMDAN()
        self.imfs = None
        self.residue = None

    def decompose(self, signal, method='ceemdan', **kwargs):
        """
        对信号进行分解
        method: 'emd', 'eemd', 'ceemdan'
        """
        print(f"使用{method.upper()}方法进行信号分解...")

        if method.lower() == 'emd':
            self.imfs = self.emd(signal, **kwargs)
        elif method.lower() == 'eemd':
            self.imfs = self.eemd(signal, **kwargs)
        elif method.lower() == 'ceemdan':
            self.imfs = self.ceemdan(signal, **kwargs)
        else:
            raise ValueError("方法必须是 'emd', 'eemd' 或 'ceemdan'")

        print(f"分解完成，得到 {len(self.imfs)} 个IMF分量")
        return self.imfs

    def plot_decomposition(self, original_signal, save_path=None):
        """绘制分解结果"""
        if self.imfs is None:
            print("请先进行信号分解!")
            return

        n_imfs = len(self.imfs)
        fig, axes = plt.subplots(n_imfs + 2, 1, figsize=(12, 2.5 * (n_imfs + 2)))

        # 绘制原始信号
        axes[0].plot(original_signal, 'b', linewidth=1.5)
        axes[0].set_title('原始信号')
        axes[0].grid(True)

        # 绘制各IMF分量
        for i, imf in enumerate(self.imfs[:-1]):  # 排除残差
            axes[i + 1].plot(imf, 'g', linewidth=1)
            axes[i + 1].set_ylabel(f'IMF {i + 1}')
            axes[i + 1].grid(True)

        # 绘制残差
        axes[-1].plot(self.imfs[-1], 'r', linewidth=1.5)
        axes[-1].set_ylabel('Residue')
        axes[-1].set_xlabel('样本点')
        axes[-1].grid(True)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def get_imf_features(self):
        """获取IMF分量的特征"""
        if self.imfs is None:
            print("请先进行信号分解!")
            return None

        features = []
        for i, imf in enumerate(self.imfs):
            # 计算每个IMF的统计特征
            imf_features = {
                'imf_index': i,
                'mean': np.mean(imf),
                'std': np.std(imf),
                'energy': np.sum(imf ** 2),
                'max_amplitude': np.max(np.abs(imf)),
                'dominant_freq': _get_dominant_frequency(imf)
            }
            features.append(imf_features)

        return features

    def reconstruct_signal(self, start_imf=0, end_imf=None):
        """重构信号（可选择部分IMF）"""
        if self.imfs is None:
            print("请先进行信号分解!")
            return None

        if end_imf is None:
            end_imf = len(self.imfs)

        return np.sum(self.imfs[start_imf:end_imf], axis=0)