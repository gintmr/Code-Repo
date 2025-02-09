import matplotlib.pyplot as plt
import pandas as pd

# 读取CSV文件
df = pd.read_csv('/data05/wuxinrui/Projects/TS/TimeGPT-1/forecast_results.csv')

# 设置图表大小
plt.figure(figsize=(12, 6))

# 绘制预测值
plt.plot(df['ds'], df['TimeGPT'], label='TimeGPT', color='blue')

# 绘制置信区间
plt.fill_between(df['ds'], df['TimeGPT-lo-80'], df['TimeGPT-hi-80'], color='blue', alpha=0.2, label='80% CI')
plt.fill_between(df['ds'], df['TimeGPT-lo-90'], df['TimeGPT-hi-90'], color='red', alpha=0.2, label='90% CI')

# 添加标题和标签
plt.title('预测结果及其置信区间')
plt.xlabel('时间')
plt.ylabel('预测值')

# 显示图例
plt.legend()

# 保存图表为PNG文件
plt.savefig('forecast_result.png', dpi=300, bbox_inches='tight')

# 关闭图表，释放内存
plt.close()