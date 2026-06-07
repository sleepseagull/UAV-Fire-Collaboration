from ultralytics.utils.plotting import plot_results
import os

# 获取当前目录下的 results.csv 绝对路径
csv_file = os.path.join(os.getcwd(), 'results.csv')

print(f"🔍 正在检查文件：{csv_file}")

if os.path.exists(csv_file):
    print("✅ 文件存在，开始绘图...")
    try:
        # 核心绘图函数：直接读取 csv 并生成 images/results.png
        plot_results(csv_file)
        print("🎉 成功！图表已保存至当前目录下的 'plots.png' 或 'results.png'")
    except Exception as e:
        print(f"❌ 绘图过程中发生错误：{e}")
        print("💡 尝试备用方案：手动使用 matplotlib 绘制")
        # 如果官方函数失效，这里可以预留手动绘制逻辑，但通常 plot_results 是最稳的
else:
    print(f"❌ 错误：找不到文件 {csv_file}")