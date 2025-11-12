import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from rich.progress import track
import warnings

warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 设置随机种子
np.random.seed(42)
tf.random.set_seed(42)


class PSOOptimizer:
    """粒子群优化器"""

    def __init__(self, n_particles=10, n_iterations=20, w=0.7, c1=1.5, c2=1.5):
        self.n_particles = n_particles  # 粒子数量
        self.n_iterations = n_iterations  # 迭代次数
        self.w = w  # 惯性权重
        self.c1 = c1  # 个体学习因子
        self.c2 = c2  # 社会学习因子

    def optimize(self, objective_function, bounds):
        """
        粒子群优化主函数
        bounds: 参数边界，格式为 [(min1, max1), (min2, max2), ...]
        """
        n_dim = len(bounds)

        # 初始化粒子位置和速度
        particles_pos = np.random.uniform(
            [b[0] for b in bounds],
            [b[1] for b in bounds],
            (self.n_particles, n_dim)
        )
        particles_vel = np.random.uniform(-1, 1, (self.n_particles, n_dim))

        # 初始化个体最佳位置和适应度
        personal_best_pos = particles_pos.copy()
        personal_best_fitness = np.array([float('inf')] * self.n_particles)

        # 初始化全局最佳
        global_best_pos = None
        global_best_fitness = float('inf')

        # 记录优化过程
        convergence_curve = []

        print("开始粒子群优化...")
        for iteration in track(range(self.n_iterations), "粒子群优化...", transient=True):
            for i in track(range(self.n_particles), "粒子群优化...(1/2)", transient=True):
                # 计算适应度
                fitness = objective_function(particles_pos[i])

                # 更新个体最佳
                if fitness < personal_best_fitness[i]:
                    personal_best_fitness[i] = fitness
                    personal_best_pos[i] = particles_pos[i].copy()

                # 更新全局最佳
                if fitness < global_best_fitness:
                    global_best_fitness = fitness
                    global_best_pos = particles_pos[i].copy()

            # 更新粒子速度和位置
            for i in track(range(self.n_particles), "粒子群优化...(2/2)", transient=True):
                r1, r2 = np.random.random(), np.random.random()

                # 速度更新
                cognitive_velocity = self.c1 * r1 * (personal_best_pos[i] - particles_pos[i])
                social_velocity = self.c2 * r2 * (global_best_pos - particles_pos[i])
                particles_vel[i] = self.w * particles_vel[i] + cognitive_velocity + social_velocity

                # 位置更新
                particles_pos[i] = particles_pos[i] + particles_vel[i]

                # 边界处理
                for dim in range(n_dim):
                    if particles_pos[i, dim] < bounds[dim][0]:
                        particles_pos[i, dim] = bounds[dim][0]
                    elif particles_pos[i, dim] > bounds[dim][1]:
                        particles_pos[i, dim] = bounds[dim][1]

            convergence_curve.append(global_best_fitness)

            if (iteration + 1) % 5 == 0:
                print(f"迭代 {iteration + 1}/{self.n_iterations}, 最佳适应度: {global_best_fitness:.6f}")

        return global_best_pos, global_best_fitness, convergence_curve


class StockPredictor:
    def __init__(self):
        self.data = None
        self.scaler = MinMaxScaler()
        self.model = None
        self.best_params = None

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

    def create_dataset(self, lookback_days=60):
        """创建时间序列数据集"""
        # 使用收盘价
        prices = self.data['收盘Close'].values.reshape(-1, 1)

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
        # 参数解码
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
            # 如果出现错误，返回一个很大的损失值
            return float('inf')

    def optimize_hyperparameters(self, lookback_days=30):
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
        # [lstm_units1, lstm_units2, dropout_rate, learning_rate]
        bounds = [
            (20, 100),  # 第一层LSTM单元数
            (10, 50),  # 第二层LSTM单元数
            (0.1, 0.5),  # Dropout率
            (0.0001, 0.01)  # 学习率
        ]

        # 创建PSO优化器
        pso = PSOOptimizer(n_particles=4, n_iterations=4)

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

        # 训练模型
        history = self.model.fit(
            X_train, y_train,
            batch_size=32,
            epochs=epochs,
            validation_split=0.2,
            verbose=1,
            callbacks=[
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

    def plot_results(self, y_true, y_pred, dates):
        """绘制结果"""
        plt.figure(figsize=(15, 10))

        # 子图1: 预测结果对比
        plt.subplot(2, 2, 1)
        plt.plot(dates, y_true, label='真实指数', color='blue', alpha=0.7, linewidth=2)
        plt.plot(dates, y_pred, label='预测指数', color='red', alpha=0.7, linewidth=2)
        plt.title('中证流通指数预测结果 (PSO优化后)')
        plt.xlabel('时间')
        plt.ylabel('指数值')
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)

        # 子图2: 预测误差
        plt.subplot(2, 2, 2)
        errors = y_true.flatten() - y_pred.flatten()
        plt.plot(dates, errors, color='green', alpha=0.7)
        plt.title('预测误差')
        plt.xlabel('时间')
        plt.ylabel('误差值')
        plt.grid(True)
        plt.xticks(rotation=45)

        # 子图3: 散点图
        plt.subplot(2, 2, 3)
        plt.scatter(y_true, y_pred, alpha=0.6)
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
        plt.xlabel('真实值')
        plt.ylabel('预测值')
        plt.title('真实值 vs 预测值')
        plt.grid(True)

        # 子图4: 误差分布直方图
        plt.subplot(2, 2, 4)
        plt.hist(errors, bins=30, alpha=0.7, color='orange', edgecolor='black')
        plt.title('预测误差分布')
        plt.xlabel('误差值')
        plt.ylabel('频次')
        plt.grid(True)

        plt.tight_layout()
        plt.show()


# 主程序
def main(file_path):
    # 创建预测器
    predictor = StockPredictor()

    # 1. 从CSV文件加载数据
    data = predictor.load_data_from_csv(file_path)

    # 显示数据基本信息
    print("\n数据基本信息:")
    print(data[['开盘Open', '最高High', '最低Low', '收盘Close']].describe())

    # 2. 使用PSO优化超参数
    best_params = predictor.optimize_hyperparameters(lookback_days=30)

    # 3. 使用优化后的参数训练最终模型
    X_test, y_test, history = predictor.train_final_model(lookback_days=30, epochs=100)

    # 4. 评估模型
    y_test_actual, y_pred_actual, metrics = predictor.evaluate_model(X_test, y_test)

    # 5. 可视化结果
    test_dates = data.index[-len(y_test_actual):]
    predictor.plot_results(y_test_actual, y_pred_actual, test_dates)

    return predictor, metrics, data, best_params


# 与基准模型对比
def baseline_comparison(data, best_params):
    """与基准模型对比"""
    # 简单LSTM基准（不使用PSO优化）
    print("\n训练基准LSTM模型进行比较...")

    predictor_baseline = StockPredictor()
    predictor_baseline.data = data

    # 使用固定参数
    fixed_params = [50, 25, 0.2, 0.001]  # 常用默认参数

    X_train, X_test, y_train, y_test = predictor_baseline.create_dataset(30)
    predictor_baseline.model = predictor_baseline.build_lstm_model(fixed_params, 30)

    # 训练基准模型
    predictor_baseline.model.fit(
        X_train, y_train,
        batch_size=32,
        epochs=80,
        validation_split=0.2,
        verbose=0
    )

    # 评估基准模型
    _, _, baseline_metrics = predictor_baseline.evaluate_model(X_test, y_test)

    # 对比结果
    print("\n" + "=" * 60)
    print("PSO优化 vs 基准LSTM 对比结果")
    print("=" * 60)

    # 重新计算PSO模型指标用于对比
    predictor_pso = StockPredictor()
    predictor_pso.data = data
    predictor_pso.best_params = best_params
    X_train, X_test, y_train, y_test = predictor_pso.create_dataset(30)
    predictor_pso.model = predictor_pso.build_lstm_model(best_params, 30)
    predictor_pso.model.fit(X_train, y_train, batch_size=32, epochs=80, validation_split=0.2, verbose=0)
    _, _, pso_metrics = predictor_pso.evaluate_model(X_test, y_test)

    print(f"{'指标':<10} {'PSO优化':<15} {'基准LSTM':<15} {'提升百分比':<15}")
    print("-" * 60)
    for metric in ['MSE', 'RMSE', 'MAE']:
        pso_val = pso_metrics[metric]
        base_val = baseline_metrics[metric]
        improvement = ((base_val - pso_val) / base_val) * 100
        print(f"{metric:<10} {pso_val:<15.6f} {base_val:<15.6f} {improvement:>10.2f}%")

    r2_improvement = (pso_metrics['R2'] - baseline_metrics['R2']) * 100
    print(f"{'R2':<10} {pso_metrics['R2']:<15.6f} {baseline_metrics['R2']:<15.6f} {r2_improvement:>10.2f}%")

    return baseline_metrics


if __name__ == "__main__":
    # 运行PSO优化的LSTM模型
    predictor, metrics, data, best_params = main("../data/000902perf.csv")

    # 与基准模型对比
    baseline_metrics = baseline_comparison(data, best_params)