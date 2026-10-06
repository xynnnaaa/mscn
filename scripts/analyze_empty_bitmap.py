import os
import struct
import csv
import numpy as np
import matplotlib.pyplot as plt

# ======== Configuration Region ========
QUERY_CSV = "/data2/xuyining/learnedcardinalities/data/zipf/test.csv"          # Path to your test.csv
CARDINALITY_CSV = "/data2/xuyining/learnedcardinalities/data/zipf/results/bitmap.csv"  # Result CSV (pred,true)
BITMAP_FILE = "/data2/xuyining/learnedcardinalities/data/zipf/test-500.bitmaps"       # Binary bitmap file
SAMPLE_SIZE = 500  # Bitmap size

# ======== Plotting Configuration ========
PLOT_NUM_BINS = 50         # Total number of bins
PLOT_BIN_TYPE = 'log'
OUTPUT_IMAGE_NAME = "empty_bitmap_distribution.png"

# ======================================

def load_large_empty_bitmap_indices(query_csv_path, bitmap_path, sample_size):
    """
    Streams query CSV and binary bitmaps simultaneously to find indices where at least one table has an empty bitmap.
    """
    bytes_per_table = int(np.ceil(sample_size / 8))
    target_indices = []
    
    if not os.path.exists(query_csv_path) or not os.path.exists(bitmap_path):
        raise FileNotFoundError("Input files missing. Check your paths.")
    
    query_idx = 0
    with open(query_csv_path, 'r') as f_csv, open(bitmap_path, 'rb') as f_bit:
        for line in f_csv:
            line = line.strip()
            if not line:
                continue
            
            parts = line.split('#')
            tables_raw = parts[0].split(',')
            tables_order = []
            for t_entry in tables_raw:
                t_entry = t_entry.strip()
                if t_entry:
                    real_table_name = t_entry.split(' ')[0]
                    tables_order.append(real_table_name)
            
            num_tables_bytes = f_bit.read(4)
            if not num_tables_bytes:
                break
            num_tables = struct.unpack('<I', num_tables_bytes)[0]
            
            has_empty_bitmap = False
            for table_name in tables_order:
                bitmap_bytes = f_bit.read(bytes_per_table)
                is_bitmap_empty = all(b == 0 for b in bitmap_bytes)
                
                if is_bitmap_empty:
                    has_empty_bitmap = True
            
            if has_empty_bitmap:
                target_indices.append(query_idx)
            query_idx += 1
            
    print(f"Scan complete. Total queries: {query_idx}, Filtered target queries: {len(target_indices)}")
    return set(target_indices)


def load_and_filter_csv(csv_path, target_indices):
    """Filter pred and true lines based on matched indices."""
    preds, trues = [], []
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        for idx, row in enumerate(reader):
            if idx in target_indices and len(row) >= 2:
                try:
                    preds.append(float(row[0]))
                    trues.append(float(row[1]))
                except ValueError:
                    continue 
    return np.array(preds), np.array(trues)


def plot_distribution(preds, trues, num_bins, bin_type):
    """
    Generates a side-by-side bar chart showing the distribution of True vs. Predicted cardinalities.
    """
    if len(preds) == 0:
        print("[Error] No data to plot.")
        return
        
    print(f"\nGenerating side-by-side bar chart ({bin_type} bins, count={num_bins}) ...")
    
    # Establish data bounds across both arrays (prevent log(0) issues)
    min_val = max(1, min(np.min(preds), np.min(trues)))
    max_val = max(1, max(np.max(preds), np.max(trues)))
    
    # Generate bin boundaries based on type
    if bin_type == 'log':
        bin_edges = np.logspace(np.log10(min_val), np.log10(max_val), num_bins + 1)
    else:
        bin_edges = np.linspace(min_val, max_val, num_bins + 1)
        
    # Calculate frequencies for each bin
    true_counts, _ = np.histogram(trues, bins=bin_edges)
    pred_counts, _ = np.histogram(preds, bins=bin_edges)
    
    # Create the side-by-side bar chart
    plt.figure(figsize=(14, 6.5))
    x = np.arange(num_bins)
    width = 0.4  # Width of each individual bar
    
    # Plot True and Pred side by side
    plt.bar(x - width/2, true_counts, width, label='True Cardinality Counts', color='#4C72B0', alpha=0.85)
    plt.bar(x + width/2, pred_counts, width, label='Predicted Cardinality Counts (Empty Bitmap)', color='#DD8452', alpha=0.85)
    
    # Label formatting for X axis (Display around 10 clean scale milestones to avoid text crowding)
    tick_spacing = max(1, num_bins // 10)
    tick_indices = np.arange(0, num_bins, tick_spacing)
    
    tick_labels = []
    for idx in tick_indices:
        low = bin_edges[idx]
        if bin_type == 'log':
            tick_labels.append(f"10^{int(np.log10(low))}" if low >= 10 else f"{int(low)}")
        else:
            tick_labels.append(f"{int(low)}")
            
    plt.xticks(tick_indices, tick_labels, rotation=0)
    
    # Titles and Labels
    plt.xlabel('Cardinality Scale (Bins Grouped by Magnitude)' if bin_type == 'log' else 'Cardinality Value Range')
    plt.ylabel('Number of Queries (Frequency)')
    plt.title('Distribution Comparison: True vs. MSCN Predicted Cardinality under Empty Bitmaps', fontsize=13, fontweight='bold', pad=15)
    
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.legend(fontsize=11, loc='upper right')
    
    # Save chart
    plt.savefig(OUTPUT_IMAGE_NAME, dpi=300, bbox_inches='tight')
    print(f"Graph successfully saved as: {OUTPUT_IMAGE_NAME}")


def main():
    try:
        target_indices = load_large_empty_bitmap_indices(
            QUERY_CSV, BITMAP_FILE, SAMPLE_SIZE
        )
        if not target_indices:
            print("No matching queries found.")
            return
            
        preds, trues = load_and_filter_csv(CARDINALITY_CSV, target_indices)
        
        plot_distribution(preds, trues, PLOT_NUM_BINS, PLOT_BIN_TYPE)
        
    except Exception as e:
        print(f"\n[Error] Failed to execute: {e}")

if __name__ == "__main__":
    main()