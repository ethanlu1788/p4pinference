import pandas as pd
import matplotlib.pyplot as plt
import ast

def plot_full_performance(csv_filename, window_size=50):
    """
    Plots smoothed CPU (%), FPS, and RAM (MB) usage over time for each model.
    RAM plots actual process memory usage in MB.
    """
    try:
        df = pd.read_csv(csv_filename)
    except FileNotFoundError:
        print(f"Error: File '{csv_filename}' not found.")
        return

    fig, axes = plt.subplots(3, 1, figsize=(15, 18), sharex=True)
    fig.suptitle('Comparative Performance Metrics Over Time', fontsize=20)
    ax1, ax2, ax3 = axes

    for index, row in df.iterrows():
        model_name = row['model_name']

        # --- CPU ---
        try:
            cpu_log = ast.literal_eval(row['cpu_log'])
            if cpu_log:
                timestamps, values = zip(*cpu_log)
                relative_times = [t - timestamps[0] for t in timestamps]
                smoothed_cpu = pd.Series(values).rolling(window=window_size, min_periods=1).mean()
                ax1.plot(relative_times, smoothed_cpu, label=f'{model_name} CPU')
        except Exception:
            pass

        # --- FPS ---
        try:
            fps_log = ast.literal_eval(row['fps_log'])
            if fps_log:
                timestamps, values = zip(*fps_log)
                relative_times = [t - timestamps[0] for t in timestamps]
                smoothed_fps = pd.Series(values).rolling(window=window_size, min_periods=1).mean()
                ax2.plot(relative_times, smoothed_fps, label=f'{model_name} FPS')
        except Exception:
            pass

        # --- RAM: Plot process memory usage in MB ---
        try:
            ram_log = ast.literal_eval(row['ram_log'])
            if ram_log:
                timestamps, ram_mb = zip(*ram_log)
                relative_times = [t - timestamps[0] for t in timestamps]
                smoothed_ram = pd.Series(ram_mb).rolling(window=window_size, min_periods=1).mean()
                ax3.plot(relative_times, smoothed_ram, label=f'{model_name} RAM (MB)')
        except Exception:
            pass

    ax1.set_title('Smoothed CPU Usage (%)')
    ax1.set_ylabel('CPU (%)')
    ax1.grid(True, linestyle='--'); ax1.legend()
    ax1.set_ylim(0, 100)

    ax2.set_title('Smoothed FPS')
    ax2.set_ylabel('Frames Per Second')
    ax2.grid(True, linestyle='--'); ax2.legend()

    ax3.set_title('Smoothed RAM Usage (MB)')
    ax3.set_xlabel('Time (seconds)')
    ax3.set_ylabel('RAM Usage (MB)')
    ax3.grid(True, linestyle='--'); ax3.legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.show()

    print("\n--- Benchmark Summary ---")
    summary_cols = ['model_name', 'iterations', 'overall_fps', 'avg_inference_fps', 'peak_ram_mb']
    print(df[summary_cols].to_string(index=False))

if __name__ == "__main__":
    CSV_FILE = "full_video_benchmark.csv"
    plot_full_performance(CSV_FILE, window_size=50)
