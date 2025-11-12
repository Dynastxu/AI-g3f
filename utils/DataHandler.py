import pandas as pd
from datetime import datetime

def convert_date_format(input_file, output_file):
    """
    将CSV文件中的日期格式从yyyymmdd转换为yyyy-mm-dd

    参数:
    input_file: 输入CSV文件路径
    output_file: 输出CSV文件路径
    """
    # 读取CSV文件
    df = pd.read_csv(input_file, header=0)

    # 获取第一列的列名
    first_column = df.columns[0]

    # 将第一列的日期格式从yyyymmdd转换为yyyy-mm-dd
    df[first_column] = pd.to_datetime(df[first_column], format='%Y%m%d').dt.strftime('%Y-%m-%d')

    # 保存到新的CSV文件
    df.to_csv(output_file, index=False, header=True)

    print(f"日期格式转换完成，已保存到 {output_file}")

if __name__ == "__main__":
    input_file = "../data/000902perf.csv"
    output_file = "../data/000902perf.csv"

    convert_date_format(input_file, output_file)
