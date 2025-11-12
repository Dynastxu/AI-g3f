import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 设置随机种子
np.random.seed(42)
tf.random.set_seed(42)


class StockPredictor:
    def __init__(self):
        self.data = None
        self.scaler = MinMaxScaler()
        self.model = None

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
        print(f"数据列名: {self.data.columns.tolist()}")

        return self.data

    def preprocess_data(self, lookback_days=60):
        """数据预处理"""
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

    def build_lstm_model(self, lookback_days=60):
        """构建LSTM模型"""
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(lookback_days, 1)),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25),
            Dense(1)
        ])

        model.compile(optimizer='adam', loss='mean_squared_error')
        self.model = model
        return model

    def train_model(self, X_train, y_train, epochs=50, batch_size=32):
        """训练模型"""
        history = self.model.fit(
            X_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_split=0.1,
            verbose=1
        )
        return history

    def predict(self, X_test):
        """预测"""
        predictions = self.model.predict(X_test)
        return predictions

    def evaluate_model(self, y_true, y_pred):
        """评估模型"""
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        print(f"MSE: {mse:.6f}")
        print(f"RMSE: {rmse:.6f}")
        print(f"MAE: {mae:.6f}")
        print(f"R²: {r2:.6f}")

        return {'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'R2': r2}

    def plot_results(self, y_true, y_pred, dates):
        """绘制结果"""
        plt.figure(figsize=(15, 10))

        # 子图1: 预测结果对比
        plt.subplot(2, 2, 1)
        plt.plot(dates, y_true, label='真实指数', color='blue', alpha=0.7, linewidth=2)
        plt.plot(dates, y_pred, label='预测指数', color='red', alpha=0.7, linewidth=2)
        plt.title('中证流通指数预测结果')
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

        # 子图4: 原始数据走势
        plt.subplot(2, 2, 4)
        plt.plot(self.data.index, self.data['收盘Close'], color='purple', alpha=0.7)
        plt.title('中证流通指数历史走势')
        plt.xlabel('时间')
        plt.ylabel('指数值')
        plt.grid(True)
        plt.xticks(rotation=45)

        plt.tight_layout()
        plt.show()


# 主程序
def main():
    # 创建预测器
    predictor = StockPredictor()

    # 1. 从CSV文件加载数据
    # 请将文件路径替换为你的实际CSV文件路径
    file_path = "../data/000902perf.csv"  # 例如: "中证流通指数.csv"
    data = predictor.load_data_from_csv(file_path)

    # 显示数据基本信息
    print("\n数据基本信息:")
    print(data[['开盘Open', '最高High', '最低Low', '收盘Close']].describe())

    # 2. 数据预处理
    X_train, X_test, y_train, y_test = predictor.preprocess_data(lookback_days=30)

    # 3. 构建模型
    model = predictor.build_lstm_model(lookback_days=30)
    print("\n模型结构:")
    model.summary()

    # 4. 训练模型
    print("\n开始训练模型...")
    history = predictor.train_model(X_train, y_train, epochs=50, batch_size=32)

    # 5. 预测
    print("\n进行预测...")
    y_pred = predictor.predict(X_test)

    # 6. 反归一化
    y_test_actual = predictor.scaler.inverse_transform(y_test.reshape(-1, 1))
    y_pred_actual = predictor.scaler.inverse_transform(y_pred)

    # 7. 评估
    print("\n模型评估结果:")
    metrics = predictor.evaluate_model(y_test_actual, y_pred_actual)

    # 8. 可视化
    test_dates = data.index[-len(y_test_actual):]
    predictor.plot_results(y_test_actual, y_pred_actual, test_dates)

    return predictor, metrics, data


# 简化的对比实验
def baseline_comparison(data):
    """与简单基准模型对比"""
    prices = data['收盘Close'].values

    # 简单移动平均预测
    lookback = 30
    split_idx = int(0.8 * len(prices))

    test_prices = prices[split_idx + lookback:]
    predictions = []

    for i in range(len(test_prices)):
        if i < lookback:
            window = prices[split_idx:split_idx + i]
        else:
            window = test_prices[i - lookback:i]
        pred = np.mean(window)
        predictions.append(pred)

    predictions = np.array(predictions)

    # 评估
    mse = mean_squared_error(test_prices, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(test_prices, predictions)
    r2 = r2_score(test_prices, predictions)

    print("\n移动平均基准模型结果:")
    print(f"MSE: {mse:.6f}")
    print(f"RMSE: {rmse:.6f}")
    print(f"MAE: {mae:.6f}")
    print(f"R²: {r2:.6f}")

    return {'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'R2': r2}


if __name__ == "__main__":
    # 运行LSTM模型
    predictor, metrics, data = main()

    # 运行基准模型对比
    baseline_metrics = baseline_comparison(data)

    # 对比结果
    print("\n" + "=" * 50)
    print("模型对比结果:")
    print("=" * 50)
    print(f"{'指标':<10} {'LSTM模型':<15} {'移动平均':<15} {'提升百分比':<15}")
    print("-" * 50)
    for metric in ['MSE', 'RMSE', 'MAE']:
        lstm_val = metrics[metric]
        base_val = baseline_metrics[metric]
        improvement = ((base_val - lstm_val) / base_val) * 100
        print(f"{metric:<10} {lstm_val:<15.6f} {base_val:<15.6f} {improvement:>10.2f}%")

    r2_improvement = (metrics['R2'] - baseline_metrics['R2']) * 100
    print(f"{'R2':<10} {metrics['R2']:<15.6f} {baseline_metrics['R2']:<15.6f} {r2_improvement:>10.2f}%")