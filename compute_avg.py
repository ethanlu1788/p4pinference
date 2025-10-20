#!/usr/bin/env python3
"""
Framework Performance Summary Script
Analyzes CSV benchmark data to calculate average CPU usage and temperature per framework
"""

import pandas as pd
import ast
import numpy as np


def analyze_framework_performance(csv_filename):
    """
    Analyzes benchmark CSV and calculates average CPU usage and temperature for each framework.

    Args:
        csv_filename (str): Path to the CSV file with benchmark results
    """
    try:
        df = pd.read_csv(csv_filename)
    except FileNotFoundError:
        print(f"Error: File '{csv_filename}' not found.")
        return

    print("=" * 80)
    print("FRAMEWORK PERFORMANCE SUMMARY")
    print("=" * 80)
    print(f"{'Framework':<20} {'Avg CPU (%)':<12} {'Avg Temp (°C)':<15} {'Peak Temp (°C)':<15}")
    print("-" * 80)

    results = []

    for index, row in df.iterrows():
        model_name = row['model_name']

        # Parse CPU log
        try:
            cpu_log = ast.literal_eval(row['cpu_log'])
            if cpu_log:
                # Extract CPU values (second element of each tuple)
                cpu_values = [cpu for _, cpu in cpu_log if cpu is not None]
                avg_cpu = np.mean(cpu_values) if cpu_values else 0
            else:
                avg_cpu = 0
        except (ValueError, SyntaxError, TypeError):
            avg_cpu = 0

        # Parse temperature log
        try:
            temp_log = ast.literal_eval(row['temp_log'])
            if temp_log:
                # Extract temperature values, filtering out None values
                temp_values = [temp for _, temp in temp_log if temp is not None]
                avg_temp = np.mean(temp_values) if temp_values else 0
            else:
                avg_temp = 0
        except (ValueError, SyntaxError, TypeError):
            avg_temp = 0

        # Get peak temperature from the summary column
        peak_temp = row.get('peak_temp_c', 0) if pd.notna(row.get('peak_temp_c', 0)) else 0

        # Display results
        print(f"{model_name:<20} {avg_cpu:<12.1f} {avg_temp:<15.1f} {peak_temp:<15.1f}")

        # Store for further analysis
        results.append({
            'framework': model_name,
            'avg_cpu': avg_cpu,
            'avg_temp': avg_temp,
            'peak_temp': peak_temp
        })

    print("-" * 80)

    # Additional analysis
    if results:
        print("\nRANKING BY EFFICIENCY:")
        print("-" * 40)

        # Sort by CPU usage (ascending = more efficient)
        cpu_sorted = sorted(results, key=lambda x: x['avg_cpu'])
        print("Most CPU Efficient:")
        for i, framework in enumerate(cpu_sorted[:3], 1):
            print(f"  {i}. {framework['framework']}: {framework['avg_cpu']:.1f}% CPU")

        print("\nLowest Temperature:")
        temp_sorted = sorted(results, key=lambda x: x['avg_temp'])
        for i, framework in enumerate(temp_sorted[:3], 1):
            print(f"  {i}. {framework['framework']}: {framework['avg_temp']:.1f}°C")

        print("\nHIGHEST RESOURCE USAGE:")
        print("-" * 40)
        cpu_sorted_desc = sorted(results, key=lambda x: x['avg_cpu'], reverse=True)
        print("Highest CPU Usage:")
        for i, framework in enumerate(cpu_sorted_desc[:3], 1):
            print(f"  {i}. {framework['framework']}: {framework['avg_cpu']:.1f}% CPU")

        temp_sorted_desc = sorted(results, key=lambda x: x['peak_temp'], reverse=True)
        print("\nHighest Peak Temperature:")
        for i, framework in enumerate(temp_sorted_desc[:3], 1):
            print(f"  {i}. {framework['framework']}: {framework['peak_temp']:.1f}°C")


def quick_summary(csv_filename):
    """
    Quick one-line summary per framework
    """
    try:
        df = pd.read_csv(csv_filename)
    except FileNotFoundError:
        print(f"Error: File '{csv_filename}' not found.")
        return

    print("\nQUICK SUMMARY:")
    print("=" * 60)

    for index, row in df.iterrows():
        model_name = row['model_name']

        # Get basic metrics from summary columns
        fps = row.get('overall_fps', 0)
        latency = row.get('avg_latency_ms', 0)
        peak_temp = row.get('peak_temp_c', 0)

        # Parse CPU for average
        try:
            cpu_log = ast.literal_eval(row['cpu_log'])
            avg_cpu = np.mean([cpu for _, cpu in cpu_log if cpu is not None]) if cpu_log else 0
        except:
            avg_cpu = 0

        print(f"{model_name}: {fps:.1f} FPS | {latency:.1f}ms | {avg_cpu:.1f}% CPU | {peak_temp:.1f}°C")


if __name__ == "__main__":
    CSV_FILE = "hailo_benchmark_results.csv"  # Your attached CSV file

    analyze_framework_performance(CSV_FILE)
    quick_summary(CSV_FILE)
