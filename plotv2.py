import pandas as pd
import matplotlib.pyplot as plt
import ast


def plot_full_performance(csv_filename, window_size=50):
    """
    Plots smoothed CPU (%), FPS, RAM (%), Temperature (°C), Power (W), and Latency (ms) over time.
    Creates six separate plots.
    Skips TFLite models in the plots.
    """
    try:
        df = pd.read_csv(csv_filename)
    except FileNotFoundError:
        print(f"Error: File '{csv_filename}' not found.")
        return

    # --- Plot 1: CPU Usage ---
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    fig1.suptitle('CPU Usage Over Time', fontsize=16)

    for index, row in df.iterrows():
        model_name = row['model_name']
        if 'tflite' in model_name.lower():
            continue

        try:
            cpu_log = ast.literal_eval(row['cpu_log'])
            if cpu_log:
                timestamps, values = zip(*cpu_log)
                relative_times = [t - timestamps[0] for t in timestamps]
                smoothed_cpu = pd.Series(values).rolling(window=window_size, min_periods=1).mean()
                ax1.plot(relative_times, smoothed_cpu, label=f'{model_name}')
        except Exception:
            pass

    ax1.set_xlabel('Time (seconds)')
    ax1.set_ylabel('CPU Usage (%)')
    ax1.set_ylim(0, 100)
    ax1.grid(True, linestyle='--')
    ax1.legend()
    plt.tight_layout()
    plt.show()

    # --- Plot 2: FPS ---
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    fig2.suptitle('FPS Over Time', fontsize=16)

    for index, row in df.iterrows():
        model_name = row['model_name']
        if 'tflite' in model_name.lower():
            continue

        try:
            fps_log = ast.literal_eval(row['fps_log'])
            if fps_log:
                timestamps, values = zip(*fps_log)
                relative_times = [t - timestamps[0] for t in timestamps]
                smoothed_fps = pd.Series(values).rolling(window=window_size, min_periods=1).mean()
                ax2.plot(relative_times, smoothed_fps, label=f'{model_name}')
        except Exception:
            pass

    ax2.set_xlabel('Time (seconds)')
    ax2.set_ylabel('Frames Per Second')
    ax2.grid(True, linestyle='--')
    ax2.legend()
    plt.tight_layout()
    plt.show()

    # --- Plot 3: RAM Usage ---
    fig3, ax3 = plt.subplots(figsize=(12, 6))
    fig3.suptitle('RAM Usage Over Time', fontsize=16)

    for index, row in df.iterrows():
        model_name = row['model_name']
        if 'tflite' in model_name.lower():
            continue

        try:
            ram_log = ast.literal_eval(row['ram_log'])
            if ram_log:
                timestamps, ram_mb = zip(*ram_log)
                relative_times = [t - timestamps[0] for t in timestamps]
                smoothed_ram = pd.Series(ram_mb).rolling(window=window_size, min_periods=1).mean()
                ax3.plot(relative_times, smoothed_ram, label=f'{model_name}')
        except Exception:
            pass

    ax3.set_xlabel('Time (seconds)')
    ax3.set_ylabel('RAM Usage (%)')
    ax3.grid(True, linestyle='--')
    ax3.legend()
    plt.tight_layout()
    plt.show()

    # --- Plot 4: Temperature ---
    fig4, ax4 = plt.subplots(figsize=(12, 6))
    fig4.suptitle('CPU Temperature Over Time', fontsize=16)

    for index, row in df.iterrows():
        model_name = row['model_name']
        if 'tflite' in model_name.lower():
            continue

        try:
            temp_log = ast.literal_eval(row['temp_log'])
            if temp_log:
                valid_temps = [(t, temp) for t, temp in temp_log if temp is not None]
                if valid_temps:
                    timestamps, values = zip(*valid_temps)
                    relative_times = [t - timestamps[0] for t in timestamps]
                    smoothed_temp = pd.Series(values).rolling(window=window_size, min_periods=1).mean()
                    ax4.plot(relative_times, smoothed_temp, label=f'{model_name}')
        except Exception:
            pass

    ax4.set_xlabel('Time (seconds)')
    ax4.set_ylabel('Temperature (°C)')
    ax4.grid(True, linestyle='--')
    ax4.legend()
    plt.tight_layout()
    plt.show()

    # --- Plot 5: Power Usage ---
    fig5, ax5 = plt.subplots(figsize=(12, 6))
    fig5.suptitle('Power Consumption Over Time', fontsize=16)

    for index, row in df.iterrows():
        model_name = row['model_name']
        if 'tflite' in model_name.lower():
            continue

        try:
            power_log = ast.literal_eval(row['power_log'])
            if power_log:
                valid_power = [(t, power) for t, power in power_log if power is not None]
                if valid_power:
                    timestamps, values = zip(*valid_power)
                    relative_times = [t - timestamps[0] for t in timestamps]
                    smoothed_power = pd.Series(values).rolling(window=window_size, min_periods=1).mean()
                    ax5.plot(relative_times, smoothed_power, label=f'{model_name}')
        except Exception:
            pass

    ax5.set_xlabel('Time (seconds)')
    ax5.set_ylabel('Power Consumption (W)')
    ax5.grid(True, linestyle='--')
    ax5.legend()
    plt.tight_layout()
    plt.show()

    # --- Plot 6: Latency ---
    fig6, ax6 = plt.subplots(figsize=(12, 6))
    fig6.suptitle('Inference Latency Over Time', fontsize=16)

    for index, row in df.iterrows():
        model_name = row['model_name']
        if 'tflite' in model_name.lower():
            continue

        try:
            latency_log = ast.literal_eval(row['latency_log'])
            if latency_log:
                timestamps, values = zip(*latency_log)  # values in ms
                relative_times = [t - timestamps[0] for t in timestamps]
                smoothed_latency = pd.Series(values).rolling(window=window_size, min_periods=1).mean()
                ax6.plot(relative_times, smoothed_latency, label=f'{model_name}')
        except Exception:
            pass

    ax6.set_xlabel('Time (seconds)')
    ax6.set_ylabel('Latency (ms)')
    ax6.grid(True, linestyle='--')
    ax6.legend()
    plt.tight_layout()
    plt.show()

    # Print summary
    print("\n--- Benchmark Summary ---")
    summary_cols = [
        'model_name', 'iterations', 'overall_fps', 'avg_latency_ms',
        'p95_latency_ms', 'peak_ram_mb'
    ]

    if 'peak_temp_c' in df.columns:
        summary_cols.append('peak_temp_c')
    if 'avg_power_w' in df.columns:
        summary_cols.append('avg_power_w')

    print(df[summary_cols].to_string(index=False))


if __name__ == "__main__":
    CSV_FILE = "pi4_benchmark.csv"
    plot_full_performance(CSV_FILE, window_size=50)
