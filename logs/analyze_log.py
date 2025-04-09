import os
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator
import argparse
import pandas as pd

def load_scalars_from_event_file(event_file, scalar_tags):
    print(f"🟢 加载事件文件: {event_file}")
    ea = event_accumulator.EventAccumulator(event_file)
    ea.Reload()

    data = {tag: [] for tag in scalar_tags}
    steps = []

    for tag in scalar_tags:
        if tag not in ea.Tags().get('scalars', []):
            print(f"⚠️ 标量 '{tag}' 未找到，跳过...")
            continue
        events = ea.Scalars(tag)
        data[tag] = [e.value for e in events]
        if not steps:
            steps.extend([e.step for e in events])

    df = pd.DataFrame(data)
    df['step'] = steps
    return df

def plot_scalars(df, scalar_tags, save_path=None):
    plt.figure(figsize=(10, 6))
    for tag in scalar_tags:
        if tag in df:
            plt.plot(df['step'], df[tag], label=tag)
    plt.xlabel("Step")
    plt.ylabel("Value")
    plt.title("Training Log Scalars")
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
        print(f"📊 图已保存至 {save_path}")
    else:
        plt.show()

def find_event_file(log_dir):
    for root, dirs, files in os.walk(log_dir):
        for fname in files:
            if fname.startswith("events.out"):
                return os.path.join(root, fname)
    raise FileNotFoundError(f"未在 {log_dir} 中找到事件文件.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', type=str, default='logs/uav_rl', help='TensorBoard 日志路径')
    parser.add_argument('--save_csv', action='store_true', help='是否保存为 CSV 文件')
    parser.add_argument('--save_plot', action='store_true', help='是否保存为图片')
    args = parser.parse_args()

    scalar_tags = ['reward/mean', 'loss/total', 'transmission/ratio']
    event_file = find_event_file(args.logdir)
    df = load_scalars_from_event_file(event_file, scalar_tags)

    if args.save_csv:
        csv_path = os.path.join(args.logdir, 'scalars.csv')
        df.to_csv(csv_path, index=False)
        print(f"✅ 已保存 CSV 至：{csv_path}")

    plot_path = os.path.join(args.logdir, 'scalars.png') if args.save_plot else None
    plot_scalars(df, scalar_tags, save_path=plot_path)
