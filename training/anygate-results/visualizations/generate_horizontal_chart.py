#!/usr/bin/env python3
"""
Generate horizontal bug combination metrics chart for full-page LaTeX display
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Load the metrics data
with open('comprehensive_test_metrics.json', 'r') as f:
    metrics = json.load(f)

bug_combo_metrics = metrics['bug_combination_metrics']

# Prepare data
combinations = list(bug_combo_metrics.keys())
precision = [bug_combo_metrics[combo]['precision'] * 100 for combo in combinations]
recall = [bug_combo_metrics[combo]['recall'] * 100 for combo in combinations]
f1 = [bug_combo_metrics[combo]['f1'] * 100 for combo in combinations]
counts = [bug_combo_metrics[combo]['count'] for combo in combinations]

# Clean up combination names for display
def clean_name(name):
    if name == 'no_bugs':
        return 'No bugs'
    # Replace underscores with spaces and capitalize
    parts = name.split('|')
    cleaned = [p.replace('_', ' ').title() for p in parts]
    # Replace "Corrupted Texture" with "Texture Corruption"
    cleaned = ['Texture Corruption' if p == 'Corrupted Texture' else p for p in cleaned]
    return ' + '.join(cleaned)

display_names = [clean_name(c) for c in combinations]

# Create figure - LARGE for full page
fig, ax = plt.subplots(figsize=(14, 11))  # Width x Height for full page

# Set up positions
y_pos = np.arange(len(combinations))
bar_height = 0.25

# Create horizontal bars
bars1 = ax.barh(y_pos - bar_height, precision, bar_height, 
                label='Precision', color='#3498db', alpha=0.9)
bars2 = ax.barh(y_pos, recall, bar_height, 
                label='Recall', color='#e74c3c', alpha=0.9)
bars3 = ax.barh(y_pos + bar_height, f1, bar_height, 
                label='F1 Score', color='#2ecc71', alpha=0.9)

# Add value labels on bars
def add_bar_labels(bars, values):
    for bar, value in zip(bars, values):
        width = bar.get_width()
        ax.text(width + 1.5, bar.get_y() + bar.get_height()/2,
                f'{value:.1f}',
                ha='left', va='center', fontsize=8, fontweight='bold')

add_bar_labels(bars1, precision)
add_bar_labels(bars2, recall)
add_bar_labels(bars3, f1)

# # Add count labels at the end
# for i, (count, pos) in enumerate(zip(counts, y_pos)):
#     ax.text(107, pos, f'n={count}',
#             ha='left', va='center', fontsize=7, 
#             style='italic', color='gray')

# Customize axes
ax.set_yticks(y_pos)
ax.set_yticklabels(display_names, fontsize=10)
ax.set_xlabel('Score (%)', fontsize=12, fontweight='bold')
ax.set_xlim(0, 115)  # Extra space for labels

# Add grid
ax.grid(axis='x', alpha=0.3, linestyle='--', linewidth=0.5)
ax.set_axisbelow(True)

# Legend
ax.legend(loc='upper right', fontsize=11, framealpha=0.95)

# Tight layout
plt.tight_layout()

# Save with high DPI for LaTeX
plt.savefig('bug_combination_metrics_horizontal.pdf', 
            dpi=300, bbox_inches='tight')
plt.savefig('bug_combination_metrics_horizontal.png', 
            dpi=300, bbox_inches='tight')

print("✅ Figure saved!")
print("   - bug_combination_metrics_horizontal.pdf (for LaTeX)")
print("   - bug_combination_metrics_horizontal.png (preview)")
