import matplotlib.pyplot as plt
import numpy as np
from matplotlib.hatch import Shapes, _hatch_types
from matplotlib.patches import Rectangle

# ---------------------------------------------------------
# 設定中文字型以解決 Glyph missing 警告
# ---------------------------------------------------------
# 根據你的路徑 (/Users/...) 判斷你使用的是 MacOS。
# 這裡設定了常見的中文字型優先順序：MacOS (Heiti TC, Arial Unicode MS) -> Windows (Microsoft JhengHei) -> Linux
plt.rcParams['font.sans-serif'] = ['Heiti TC', 'PingFang TC', 'Arial Unicode MS', 'Microsoft JhengHei', 'SimHei', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False # 解決更換字型後，負號無法顯示的問題

# ---------------------------------------------------------
# 1. 定義自定義紋理類別 (參考自 malithjayaweera.com)
# ---------------------------------------------------------
class SquareHatch(Shapes):
    """
    Square hatch defined by a path drawn inside [-0.5, 0.5] square.
    Identifier 's'.
    自定義的方形紋理，代號為 's'
    """
    def __init__(self, hatch, density):
        self.filled = False
        self.size = 1
        # 定義一個方形路徑 (-0.25, 0.25) 到 寬高 0.5
        self.path = Rectangle((-0.25, 0.25), 0.5, 0.5).get_path()
        # 根據字元出現次數決定密度 (例如 'ss' 比 's' 密)
        self.num_rows = (hatch.count('s')) * density
        self.shape_vertices = self.path.vertices
        self.shape_codes = self.path.codes
        Shapes.__init__(self, hatch, density)

# 2. 將自定義紋理註冊到 Matplotlib
# 注意：這是一個比較進階的操作，直接修改了 Matplotlib 的內部列表
# 檢查是否已經存在，避免重複添加
if SquareHatch not in _hatch_types:
    _hatch_types.append(SquareHatch)

def plot_hatch_patterns():
    # 基礎紋理符號列表，現在加入了我們自定義的 's' (Square)
    patterns = [
        '/', '\\', '|', '-', '+', 
        'x', 'o', 'O', '.', '*',
        's'  # <--- 新增的自定義方形紋理
    ]
    
    # 調整佈局為 3x4 (容納 11 個圖樣)
    fig, axs = plt.subplots(3, 4, figsize=(16, 12), constrained_layout=True)
    fig.suptitle('Matplotlib Standard & Custom Hatches (含自定義方形)', fontsize=20)
    
    # 扁平化 axes 陣列以便迭代
    axs = axs.flatten()
    
    for i, ax in enumerate(axs):
        if i >= len(patterns):
            ax.axis('off') # 隱藏多餘的子圖
            continue
            
        pattern = patterns[i]
        
        # 數據
        # 繪製三根柱狀圖來展示效果
        
        # Bar 1: 基礎密度
        ax.bar(1, 3, color='white', edgecolor='tab:blue', hatch=pattern, linewidth=2)
        
        # Bar 2: 加密 (Repeating the string increases density)
        # 對於 's' 來說，'ss' 會比 's' 更密
        dense_pattern = pattern * 3
        ax.bar(2, 4, color='white', edgecolor='tab:orange', hatch=dense_pattern, linewidth=2)
        
        # Bar 3: 混合
        if pattern == 's':
             # 自定義示範：方形與斜線混合
             mixed_pattern = 's/' 
             label_text = "Mixed: s/"
        elif pattern in ['.', '*', 'o', 'O']:
             mixed_pattern = pattern * 6 
             label_text = f"Density: {pattern*6}"
        else:
             mixed_pattern = pattern + 'o' 
             label_text = f"Mixed: {pattern}+o"
             
        ax.bar(3, 3, color='white', edgecolor='tab:green', hatch=mixed_pattern, linewidth=2)
        
        # 設定標題
        title_text = f"Custom: '{pattern}' (Square)" if pattern == 's' else f"Standard: '{pattern}'"
        ax.set_title(title_text, fontsize=14, fontweight='bold', 
                     color='darkred' if pattern == 's' else 'black')
        
        ax.set_xticks([1, 2, 3])
        ax.set_xticklabels(['Basic', 'Dense (x3)', 'Mixed'], fontsize=9)
        ax.set_ylim(0, 5)
        ax.set_yticks([]) 
        
        # 套用網站建議的 "Clean Look" (移除多餘邊框)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        # ax.spines['bottom'].set_color('gray')

    # 添加說明
    plt.figtext(0.5, 0.02, 
                "Note: The 's' pattern is a custom class added via matplotlib.hatch.Shapes.\n"
                "註：'s' 是透過繼承 Shapes 類別新增的自定義方形紋理 (參考 malithjayaweera.com)。",
                ha="center", fontsize=12, bbox={"facecolor":"#f0f0f0", "alpha":0.5, "pad":10})
    
    plt.show()

if __name__ == "__main__":
    plot_hatch_patterns()