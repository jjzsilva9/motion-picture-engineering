import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Constants
GT_DIR = 'ground truth'
MAP_DIR = 'map_mattes'
NN_DIR = 'nn_mattes'
OUTPUT_DIR = 'presentation_images'

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# 1. Generate Metrics Graph
# Data from the notebook results
data = {
    'Frame': [47, 48, 49],
    'MAP_Matte_MSE': [626.4, 755.7, 1473.8],
    'NN_Matte_MSE': [719.9, 979.5, 1706.8],
    'MAP_Matte_SSIM': [0.981, 0.979, 0.963],
    'NN_Matte_SSIM': [0.977, 0.973, 0.956],
    'MAP_Matte_IoU': [0.937, 0.922, 0.860],
    'NN_Matte_IoU': [0.924, 0.895, 0.831]
}
df = pd.DataFrame(data)
summary = df.mean(numeric_only=True).to_frame().T

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({'font.size': 12})

fig, axes = plt.subplots(1, 4, figsize=(20, 5))
fig.suptitle("Performance Metrics: MAP vs. U-Net", fontsize=22, y=1.05)

# SSIM
metrics_ssim = pd.DataFrame({
    'Solution': ['MAP', 'NN'],
    'SSIM': [summary['MAP_Matte_SSIM'][0], summary['NN_Matte_SSIM'][0]]
})
sns.barplot(data=metrics_ssim, x='Solution', y='SSIM', ax=axes[0], palette="viridis")
axes[0].set_title("Matte SSIM (↑)")
axes[0].set_ylim(0.9, 1.0)

# IoU
metrics_iou = pd.DataFrame({
    'Solution': ['MAP', 'NN'],
    'IoU': [summary['MAP_Matte_IoU'][0], summary['NN_Matte_IoU'][0]]
})
sns.barplot(data=metrics_iou, x='Solution', y='IoU', ax=axes[1], palette="magma")
axes[1].set_title("Matte IoU (↑)")
axes[1].set_ylim(0.7, 1.0)

# MSE
metrics_mse = pd.DataFrame({
    'Solution': ['MAP', 'NN'],
    'MSE': [summary['MAP_Matte_MSE'][0], summary['NN_Matte_MSE'][0]]
})
sns.barplot(data=metrics_mse, x='Solution', y='MSE', ax=axes[2], palette="rocket")
axes[2].set_title("Matte MSE (↓)")

# Frame-by-Frame Progression
melted_df = df.melt(id_vars=['Frame'], value_vars=['MAP_Matte_IoU', 'NN_Matte_IoU'], 
                    var_name='Solution', value_name='IoU')
melted_df['Solution'] = melted_df['Solution'].str.replace('_Matte_IoU', '')
sns.lineplot(data=melted_df, x='Frame', y='IoU', hue='Solution', marker='o', ax=axes[3])
axes[3].set_title("IoU over Time")
axes[3].set_xticks([47, 48, 49])

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'metrics_graph.png'), dpi=150)
plt.close()

# 2. Visual Comparison Montage (Frame 47)
frame = "0047"
frame5 = "00047"

gt_matte = cv2.imread(os.path.join(GT_DIR, f'Hula.Fore.ACKGT.{frame5}.png'))
map_matte = cv2.imread(os.path.join(MAP_DIR, f'Hula.{frame}.png'))
nn_matte = cv2.imread(os.path.join(NN_DIR, f'Hula.{frame}.png'))

# Add labels to images
def add_label(img, text):
    img = img.copy()
    cv2.putText(img, text, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 8)
    return img

gt_matte = add_label(gt_matte, "Ground Truth")
map_matte = add_label(map_matte, "MAP (Ours)")
nn_matte = add_label(nn_matte, "U-Net")

# Save individual images for columns
cv2.imwrite(os.path.join(OUTPUT_DIR, 'gt_output.png'), gt_matte)
cv2.imwrite(os.path.join(OUTPUT_DIR, 'map_output.png'), map_matte)
cv2.imwrite(os.path.join(OUTPUT_DIR, 'nn_output.png'), nn_matte)

# 3. Composite Comparison
gt_comp = cv2.imread(os.path.join(GT_DIR, 'composite', f'Hula.{frame}.png'))
map_comp = cv2.imread(os.path.join(MAP_DIR, 'composite', f'Hula.{frame}.png'))

cv2.imwrite(os.path.join(OUTPUT_DIR, 'map_comp.png'), map_comp)

print("Images generated successfully in presentation_images/")
