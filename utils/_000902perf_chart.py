import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


def plot_index_trend(csv_file_path, save_path=None):
    """
    绘制中证流通指数历史走势图

    Args:
        csv_file_path: CSV文件路径
        save_path: 图片保存路径（可选）
    """

    # 读取数据
    print("正在读取数据...")
    df = pd.read_csv(csv_file_path)

    # 数据预处理
    df['日期Date'] = pd.to_datetime(df['日期Date'])
    df = df.sort_values('日期Date')
    df.set_index('日期Date', inplace=True)

    print(f"数据时间范围: {df.index.min()} 到 {df.index.max()}")
    print(f"数据总量: {len(df)} 个交易日")

    # 创建图形
    plt.figure(figsize=(14, 8))

    # 绘制收盘价走势
    plt.plot(df.index, df['收盘Close'],
             color='#1f77b4', linewidth=1.5, label='收盘价')

    # 添加一些关键统计信息
    max_price = df['收盘Close'].max()
    min_price = df['收盘Close'].min()
    current_price = df['收盘Close'].iloc[-1]

    # 标记最高点和最低点
    max_date = df['收盘Close'].idxmax()
    min_date = df['收盘Close'].idxmin()

    plt.scatter(max_date, max_price, color='red', s=80, zorder=5,
                label=f'最高点: {max_price:.2f}')
    plt.scatter(min_date, min_price, color='green', s=80, zorder=5,
                label=f'最低点: {min_price:.2f}')

    # 添加当前价格标记
    plt.axhline(y=current_price, color='orange', linestyle='--', alpha=0.7,
                label=f'当前价格: {current_price:.2f}')

    # 计算移动平均线（30日）
    df['MA30'] = df['收盘Close'].rolling(window=30).mean()
    plt.plot(df.index, df['MA30'], color='red', linewidth=1, alpha=0.8,
             label='30日移动平均线')

    # 计算移动平均线（60日）
    df['MA60'] = df['收盘Close'].rolling(window=60).mean()
    plt.plot(df.index, df['MA60'], color='purple', linewidth=1, alpha=0.8,
             label='60日移动平均线')

    # 美化图形
    plt.title('图 3-1-2 中证流通指数历史走势图 (902.CSI)', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('日期', fontsize=12)
    plt.ylabel('指数值', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='lower left', frameon=True, fancybox=True, shadow=True)

    # 设置x轴日期格式
    plt.gcf().autofmt_xdate()

    plt.tight_layout()

    # 保存图片
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图片已保存至: {save_path}")

    plt.show()

    return df


def plot_detailed_analysis(df, save_path=None):
    """
    绘制详细的技术分析图表
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('中证流通指数技术分析', fontsize=16, fontweight='bold')

    # 子图1: 价格与成交量
    ax1 = axes[0, 0]
    color = 'tab:blue'
    ax1.set_xlabel('日期')
    ax1.set_ylabel('收盘价', color=color)
    ax1.plot(df.index, df['收盘Close'], color=color, linewidth=1.5)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, alpha=0.3)

    ax1_vol = ax1.twinx()
    color = 'tab:gray'
    ax1_vol.set_ylabel('成交量(万手)', color=color)
    ax1_vol.fill_between(df.index, 0, df['成交量（万手）Volume(M Shares)'],
                         alpha=0.3, color=color)
    ax1_vol.tick_params(axis='y', labelcolor=color)

    ax1.set_title('价格与成交量')

    # 子图2: 价格与移动平均线
    ax2 = axes[0, 1]
    ax2.plot(df.index, df['收盘Close'], label='收盘价', linewidth=1, alpha=0.7)
    ax2.plot(df.index, df['MA30'], label='30日均线', linewidth=1.5)
    ax2.plot(df.index, df['MA60'], label='60日均线', linewidth=1.5)
    ax2.set_title('移动平均线分析')
    ax2.set_xlabel('日期')
    ax2.set_ylabel('指数值')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 子图3: 涨跌幅分布
    ax3 = axes[1, 0]
    daily_returns = df['涨跌幅(%)Change(%)']
    ax3.hist(daily_returns, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    ax3.axvline(daily_returns.mean(), color='red', linestyle='--',
                label=f'均值: {daily_returns.mean():.2f}%')
    ax3.set_title('日涨跌幅分布')
    ax3.set_xlabel('涨跌幅(%)')
    ax3.set_ylabel('频次')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 子图4: 价格区间统计
    ax4 = axes[1, 1]
    price_ranges = [
        (df['收盘Close'].min(), df['收盘Close'].quantile(0.25)),
        (df['收盘Close'].quantile(0.25), df['收盘Close'].quantile(0.5)),
        (df['收盘Close'].quantile(0.5), df['收盘Close'].quantile(0.75)),
        (df['收盘Close'].quantile(0.75), df['收盘Close'].max())
    ]
    range_labels = ['0-25%', '25-50%', '50-75%', '75-100%']
    range_counts = []

    for low, high in price_ranges:
        count = ((df['收盘Close'] >= low) & (df['收盘Close'] < high)).sum()
        range_counts.append(count)

    colors = ['lightcoral', 'lightyellow', 'lightgreen', 'lightblue']
    bars = ax4.bar(range_labels, range_counts, color=colors, edgecolor='black')
    ax4.set_title('价格区间分布')
    ax4.set_xlabel('价格分位数区间')
    ax4.set_ylabel('交易天数')

    # 在柱状图上添加数值标签
    for bar, count in zip(bars, range_counts):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{count}天\n({count / len(df) * 100:.1f}%)',
                 ha='center', va='bottom')

    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"详细分析图已保存至: {save_path}")

    plt.show()


def plot_monthly_analysis(df, save_path=None):
    """
    绘制月度分析图表
    """
    # 提取月份信息
    df['Year'] = df.index.year
    df['Month'] = df.index.month

    # 计算月度收益率
    monthly_data = df.groupby(['Year', 'Month'])['收盘Close'].agg(['first', 'last'])
    monthly_data['Monthly_Return'] = (monthly_data['last'] / monthly_data['first'] - 1) * 100

    # 创建月度热力图
    monthly_returns = monthly_data['Monthly_Return'].unstack(level=0)

    plt.figure(figsize=(14, 8))

    # 创建热力图
    im = plt.imshow(monthly_returns.T, cmap='RdYlGn', aspect='auto',
                    vmin=-10, vmax=10)  # 设置颜色范围在-10%到+10%

    # 设置坐标轴
    months = ['1月', '2月', '3月', '4月', '5月', '6月',
              '7月', '8月', '9月', '10月', '11月', '12月']
    years = monthly_returns.columns.tolist()

    plt.xticks(range(12), months)
    plt.yticks(range(len(years)), years)
    plt.xlabel('月份')
    plt.ylabel('年份')

    # 添加颜色条
    cbar = plt.colorbar(im, pad=0.01)
    cbar.set_label('月度收益率(%)', rotation=270, labelpad=15)

    # 在每个格子中添加数值
    for i in range(len(years)):
        for j in range(12):
            if j in monthly_returns.index and years[i] in monthly_returns.columns:
                value = monthly_returns.loc[j + 1, years[i]]
                if not pd.isna(value):
                    text = plt.text(j, i, f'{value:.1f}%',
                                    ha="center", va="center",
                                    color="black" if abs(value) < 5 else "white",
                                    fontsize=8)

    plt.title('中证流通指数月度收益率热力图(%)', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"月度分析图已保存至: {save_path}")

    plt.show()

    return monthly_returns


# 主程序
if __name__ == "__main__":
    csv_file = "../data/000902perf.csv"
    output_path = "../docs/imgs/"

    try:
        # 绘制主要走势图
        df = plot_index_trend(csv_file, save_path=f"{output_path}中证流通指数历史走势图.png")

        # 绘制详细分析图
        plot_detailed_analysis(df, save_path=f"{output_path}中证流通指数技术分析.png")

        # 绘制月度分析图
        monthly_returns = plot_monthly_analysis(df, save_path=f"{output_path}中证流通指数月度分析.png")

        # 打印一些基本统计信息
        print("\n" + "=" * 50)
        print("中证流通指数基本统计信息")
        print("=" * 50)
        print(f"数据期间: {df.index.min().strftime('%Y-%m-%d')} 至 {df.index.max().strftime('%Y-%m-%d')}")
        print(f"总交易日数: {len(df):,}")
        print(f"起始价格: {df['收盘Close'].iloc[0]:.2f}")
        print(f"结束价格: {df['收盘Close'].iloc[-1]:.2f}")
        print(f"期间涨跌: {df['收盘Close'].iloc[-1] - df['收盘Close'].iloc[0]:.2f}")
        print(f"期间涨跌幅: {(df['收盘Close'].iloc[-1] / df['收盘Close'].iloc[0] - 1) * 100:.2f}%")
        print(f"最高价格: {df['收盘Close'].max():.2f}")
        print(f"最低价格: {df['收盘Close'].min():.2f}")
        print(f"平均价格: {df['收盘Close'].mean():.2f}")
        print(f"价格标准差: {df['收盘Close'].std():.2f}")
        print(f"日均成交量: {df['成交量（万手）Volume(M Shares)'].mean():.2f}万手")
        print(f"日均成交金额: {df['成交金额（亿元）Turnover'].mean():.2f}亿元")

    except FileNotFoundError:
        print(f"错误: 找不到文件 {csv_file}")
        print("请确保文件路径正确")
    except Exception as e:
        print(f"发生错误: {e}")