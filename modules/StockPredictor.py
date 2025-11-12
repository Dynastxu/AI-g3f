# StockPredictor.py (修改版)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import warnings

from modules.PSOOptimizer import PSOOptimizer
from modules.SignalDecomposer import SignalDecomposer  # 新增导入

warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 设置随机种子
np.random.seed(42)
tf.random.set_seed(42)


class StockPredictor:
    def __init__(self, use_decomposition=False, decomposition_method='ceemdan'):
        self.data = None
        self.scaler = MinMaxScaler()
        self.model = None
        self.best_params = None
        self.use_decomposition = use_decomposition
        self.decomposition_method = decomposition_method
        self.decomposer = SignalDecomposer() if use_decomposition else None
        self.imfs = None

    def load_data_from_csv(self, file_path):
        """从CSV文件加载数据"""
        print("正在加载CSV数据...")
        self.data = pd.read_csv(file_path)

        # 数据预处理
        self.data['日期Date'] = pd.to_datetime(self.data['日期Date'])
        self.data = self.data.sort_values('日期Date')
        self.data.set_index('日期Date', inplace=True)

        print(f"数据形状: {self.data.shape}")
        print(f"数据时间范围: {self.data.index.min()} 到 {self.data.index.max()}")

        return self.data

    def apply_decomposition(self):
        """应用信号分解"""
        if not self.use_decomposition:
            print("未启用信号分解")
            return None

        print(f"\n开始{self.decomposition_method.upper()}信号分解...")

        # 使用收盘价进行分解
        prices = self.data['收盘Close'].values

        # 进行信号分解
        self.imfs = self.decomposer.decompose(prices, method=self.decomposition_method)

        # 绘制分解结果
        self.decomposer.plot_decomposition(prices)

        # 显示IMF特征
        features = self.decomposer.get_imf_features()
        print("\nIMF分量特征:")
        for feature in features:
            print(f"IMF{feature['imf_index']}: 均值={feature['mean']:.4f}, "
                  f"标准差={feature['std']:.4f}, 能量={feature['energy']:.4f}")

        return self.imfs

    def create_dataset(self, lookback_days=60, selected_imfs=None):
        """创建时间序列数据集"""
        # 使用收盘价
        prices = self.data['收盘Close'].values.reshape(-1, 1)

        if self.use_decomposition and self.imfs is not None:
            print("使用分解后的信号分量进行预测...")

            if selected_imfs is None:
                # 默认使用所有高频IMF分量（排除残差）
                selected_imfs = list(range(len(self.imfs) - 1))

            # 重构选定的IMF分量
            reconstructed = self.decomposer.reconstruct_signal(
                start_imf=min(selected_imfs),
                end_imf=max(selected_imfs) + 1
            )
            prices = reconstructed.reshape(-1, 1)
            print(f"使用IMF分量 {selected_imfs} 进行预测")

        # 归一化
        scaled_prices = self.scaler.fit_transform(prices)

        # 创建时间序列数据
        X, y = [], []
        for i in range(lookback_days, len(scaled_prices)):
            X.append(scaled_prices[i - lookback_days:i, 0])
            y.append(scaled_prices[i, 0])

        X, y = np.array(X), np.array(y)

        # 划分训练集和测试集 (80% 训练, 20% 测试)
        split_idx = int(0.8 * len(X))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        # 调整LSTM输入形状
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

        return X_train, X_test, y_train, y_test

    def build_lstm_model(self, params, lookback_days=60):
        """根据参数构建LSTM模型"""
        lstm_units1 = int(params[0])  # 第一层LSTM单元数
        lstm_units2 = int(params[1])  # 第二层LSTM单元数
        dropout_rate = params[2]  # Dropout率
        learning_rate = params[3]  # 学习率

        model = Sequential([
            LSTM(lstm_units1, return_sequences=True, input_shape=(lookback_days, 1)),
            Dropout(dropout_rate),
            LSTM(lstm_units2, return_sequences=False),
            Dropout(dropout_rate),
            Dense(25),
            Dense(1)
        ])

        optimizer = Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss='mean_squared_error')

        return model

    def objective_function(self, params):
        """PSO的目标函数 - 返回验证集上的损失"""
        lookback_days = 30  # 固定时间步长

        try:
            # 构建模型
            model = self.build_lstm_model(params, lookback_days)

            # 使用部分数据进行快速验证（为了PSO效率）
            quick_train_size = min(500, len(self.X_train_quick))
            quick_val_size = min(100, len(self.X_val_quick))

            # 训练模型（使用较少轮次）
            history = model.fit(
                self.X_train_quick[:quick_train_size],
                self.y_train_quick[:quick_train_size],
                batch_size=32,
                epochs=10,
                validation_data=(
                    self.X_val_quick[:quick_val_size],
                    self.y_val_quick[:quick_val_size]
                ),
                verbose=0
            )

            # 返回验证损失
            return history.history['val_loss'][-1]

        except Exception as e:
            print(f"PSO目标函数错误: {e}")
            return float('inf')

    def optimize_hyperparameters(self, lookback_days, n_particles, n_iterations):
        """使用PSO优化超参数"""
        print("准备PSO优化数据...")

        # 准备快速验证数据
        X_train, X_test, y_train, y_test = self.create_dataset(lookback_days)

        # 从训练集中划分验证集
        val_size = int(0.2 * len(X_train))
        self.X_train_quick = X_train[:-val_size]
        self.X_val_quick = X_train[-val_size:]
        self.y_train_quick = y_train[:-val_size]
        self.y_val_quick = y_train[-val_size:]

        # 定义参数边界
        bounds = [
            (20, 100),  # 第一层LSTM单元数
            (10, 50),  # 第二层LSTM单元数
            (0.1, 0.5),  # Dropout率
            (0.0001, 0.01)  # 学习率
        ]

        # 创建PSO优化器
        pso = PSOOptimizer(n_particles, n_iterations)

        # 运行优化
        best_params, best_fitness, convergence_curve = pso.optimize(
            self.objective_function, bounds
        )

        # 保存最佳参数
        self.best_params = best_params

        print("\nPSO优化完成!")
        print(f"最佳参数: LSTM1={int(best_params[0])}, LSTM2={int(best_params[1])}, "
              f"Dropout={best_params[2]:.3f}, LR={best_params[3]:.6f}")
        print(f"最佳适应度: {best_fitness:.6f}")

        # 绘制收敛曲线
        plt.figure(figsize=(10, 6))
        plt.plot(convergence_curve, 'b-', linewidth=2)
        plt.title('PSO收敛曲线')
        plt.xlabel('迭代次数')
        plt.ylabel('适应度值 (验证损失)')
        plt.grid(True)
        plt.show()

        return best_params

    def train_final_model(self, lookback_days=30, epochs=100):
        """使用优化后的参数训练最终模型"""
        if self.best_params is None:
            print("请先运行参数优化!")
            return None

        print("使用优化参数训练最终模型...")

        # 准备完整数据
        X_train, X_test, y_train, y_test = self.create_dataset(lookback_days)

        # 构建最终模型
        self.model = self.build_lstm_model(self.best_params, lookback_days)

        print("最终模型结构:")
        self.model.summary()

        custom_logger = CustomLoggingCallback(log_interval=10)  # 每10个epoch输出一次

        # 训练模型
        history = self.model.fit(
            X_train, y_train,
            batch_size=32,
            epochs=epochs,
            validation_split=0.2,
            verbose=0,
            callbacks=[
                custom_logger,
                tf.keras.callbacks.EarlyStopping(
                    patience=15,
                    restore_best_weights=True
                )
            ]
        )

        # 绘制训练历史
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='训练损失')
        plt.plot(history.history['val_loss'], label='验证损失')
        plt.title('模型训练历史')
        plt.xlabel('轮次')
        plt.ylabel('损失')
        plt.legend()
        plt.grid(True)
        plt.show()

        return X_test, y_test, history

    def evaluate_model(self, X_test, y_test):
        """评估模型性能"""
        # 预测
        y_pred = self.model.predict(X_test)

        # 反归一化
        y_test_actual = self.scaler.inverse_transform(y_test.reshape(-1, 1))
        y_pred_actual = self.scaler.inverse_transform(y_pred)

        # 计算评估指标
        mse = mean_squared_error(y_test_actual, y_pred_actual)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test_actual, y_pred_actual)
        r2 = r2_score(y_test_actual, y_pred_actual)

        print("\n模型评估结果:")
        print(f"MSE: {mse:.6f}")
        print(f"RMSE: {rmse:.6f}")
        print(f"MAE: {mae:.6f}")
        print(f"R²: {r2:.6f}")

        return y_test_actual, y_pred_actual, {'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'R2': r2}

    def plot_comprehensive_results(self, y_true, y_pred, dates):
        """绘制综合结果"""
        fig = plt.figure(figsize=(18, 12))

        # 子图1: 预测结果对比
        plt.subplot(2, 3, 1)
        plt.plot(dates, y_true, label='真实指数', color='blue', alpha=0.7, linewidth=2)
        plt.plot(dates, y_pred, label='预测指数', color='red', alpha=0.7, linewidth=2)
        title = '中证流通指数预测结果'
        if self.use_decomposition:
            title += f' ({self.decomposition_method.upper()}+PSO+LSTM)'
        else:
            title += ' (PSO+LSTM)'
        plt.title(title)
        plt.xlabel('时间')
        plt.ylabel('指数值')
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)

        # 子图2: 预测误差
        plt.subplot(2, 3, 2)
        errors = y_true.flatten() - y_pred.flatten()
        plt.plot(dates, errors, color='green', alpha=0.7)
        plt.title('预测误差')
        plt.xlabel('时间')
        plt.ylabel('误差值')
        plt.grid(True)
        plt.xticks(rotation=45)

        # 子图3: 散点图
        plt.subplot(2, 3, 3)
        plt.scatter(y_true, y_pred, alpha=0.6)
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
        plt.xlabel('真实值')
        plt.ylabel('预测值')
        plt.title('真实值 vs 预测值')
        plt.grid(True)

        # 子图4: 误差分布直方图
        plt.subplot(2, 3, 4)
        plt.hist(errors, bins=30, alpha=0.7, color='orange', edgecolor='black')
        plt.title('预测误差分布')
        plt.xlabel('误差值')
        plt.ylabel('频次')
        plt.grid(True)

        # 子图5: 累积误差
        plt.subplot(2, 3, 5)
        cumulative_errors = np.cumsum(np.abs(errors))
        plt.plot(dates, cumulative_errors, color='purple')
        plt.title('累积绝对误差')
        plt.xlabel('时间')
        plt.ylabel('累积误差')
        plt.grid(True)
        plt.xticks(rotation=45)

        # 子图6: 相对误差百分比
        plt.subplot(2, 3, 6)
        relative_errors = (errors / y_true.flatten()) * 100
        plt.plot(dates, relative_errors, color='brown')
        plt.title('相对误差百分比')
        plt.xlabel('时间')
        plt.ylabel('相对误差(%)')
        plt.grid(True)
        plt.xticks(rotation=45)

        plt.tight_layout()
        plt.show()

class CustomLoggingCallback(tf.keras.callbacks.Callback):
    def __init__(self, log_interval=10):
        super().__init__()
        self.log_interval = log_interval

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.log_interval == 0:
            print(f"Epoch {epoch + 1}: loss={logs['loss']:.4f}, val_loss={logs['val_loss']:.4f}")

# 综合比较函数
def comprehensive_comparison(file_path):
    """综合比较不同方法的性能"""

    methods = [
        {'name': 'LSTM', 'use_decomp': False, 'method': None},
        {'name': 'EMD+LSTM', 'use_decomp': True, 'method': 'emd'},
        {'name': 'EEMD+LSTM', 'use_decomp': True, 'method': 'eemd'},
        {'name': 'CEEMDAN+LSTM', 'use_decomp': True, 'method': 'ceemdan'},
    ]

    results = {}

    for method_config in methods:
        print(f"\n{'=' * 60}")
        print(f"正在测试: {method_config['name']}")
        print(f"{'=' * 60}")

        # 创建预测器
        predictor = StockPredictor(
            use_decomposition=method_config['use_decomp'],
            decomposition_method=method_config['method']
        )

        # 加载数据
        data = predictor.load_data_from_csv(file_path)

        # 应用分解（如果启用）
        if method_config['use_decomp']:
            predictor.apply_decomposition()

        # 优化参数
        best_params = predictor.optimize_hyperparameters(30, 4, 4)

        # 训练模型
        X_test, y_test, history = predictor.train_final_model(lookback_days=30, epochs=100)

        # 评估模型
        y_test_actual, y_pred_actual, metrics = predictor.evaluate_model(X_test, y_test)

        # 保存结果
        results[method_config['name']] = {
            'metrics': metrics,
            'predictions': y_pred_actual,
            'actual': y_test_actual
        }

        # 可视化结果
        test_dates = data.index[-len(y_test_actual):]
        predictor.plot_comprehensive_results(y_test_actual, y_pred_actual, test_dates)

    # 对比所有方法
    print(f"\n{'=' * 80}")
    print("所有方法性能对比")
    print(f"{'=' * 80}")
    print(f"{'方法':<15} {'MSE':<12} {'RMSE':<12} {'MAE':<12} {'R²':<12}")
    print(f"{'-' * 80}")

    for method_name, result in results.items():
        m = result['metrics']
        print(f"{method_name:<15} {m['MSE']:<12.6f} {m['RMSE']:<12.6f} {m['MAE']:<12.6f} {m['R2']:<12.6f}")

    return results


# 主程序
def main(file_path, use_decomposition=True, decomposition_method='ceemdan'):
    # 创建预测器
    predictor = StockPredictor(
        use_decomposition=use_decomposition,
        decomposition_method=decomposition_method
    )

    # 1. 从CSV文件加载数据
    data = predictor.load_data_from_csv(file_path)

    # 显示数据基本信息
    print("\n数据基本信息:")
    print(data[['开盘Open', '最高High', '最低Low', '收盘Close']].describe())

    # 2. 应用信号分解（如果启用）
    if use_decomposition:
        predictor.apply_decomposition()

    # 3. 使用PSO优化超参数
    best_params = predictor.optimize_hyperparameters(30, 2, 2)

    # 4. 使用优化后的参数训练最终模型
    X_test, y_test, history = predictor.train_final_model(lookback_days=30, epochs=100)

    # 5. 评估模型
    y_test_actual, y_pred_actual, metrics = predictor.evaluate_model(X_test, y_test)

    # 6. 可视化结果
    test_dates = data.index[-len(y_test_actual):]
    predictor.plot_comprehensive_results(y_test_actual, y_pred_actual, test_dates)

    return predictor, metrics, data, best_params


if __name__ == "__main__":
    file_path = "../data/000902perf.csv"

    # 方式1: 运行单个方法
    predictor, metrics, data, best_params = main(
        file_path,
        use_decomposition=True,
        decomposition_method='ceemdan'
    )

    # # 方式2: 运行综合比较（比较所有方法）
    # results = comprehensive_comparison(file_path)