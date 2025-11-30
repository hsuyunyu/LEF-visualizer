import sys
import json
import os
import argparse
import re

# --- 檢查並匯入視覺化函式庫 ---
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import numpy as np
    import matplotlib.colors as mcolors
    
    PLOT_AVAILABLE = True
    
    # 設定 Matplotlib 參數以支援負號顯示
    plt.rcParams['axes.unicode_minus'] = False 

except ImportError:
    print("[錯誤] 找不到 'matplotlib' 或 'numpy'。請安裝：pip install matplotlib numpy")
    sys.exit(1)


# --- 視覺化設定與功能 ---

# PIN 類型對應的顏色 (用於 PIN 名稱文字顏色 與 圖例)
PIN_DIRECTION_COLORS_DARK = {
    'INPUT': "#72f8a3",  
    'OUTPUT': "#ffd249", 
    'INOUT': "#9fb2b6", 
    'UNKNOWN': 'grey'    
}

PIN_DIRECTION_COLORS_LIGHT = {
    'INPUT': "#72f8a3",  
    'OUTPUT': "#ffee01",
    'INOUT': "#000000",  
    'UNKNOWN': 'grey'    
}

# 預設顯示的圖層 (正規表達式列表)
# 預設顯示所有圖層 ('.*')，確保 VIA 和其他層能出現
DEFAULT_TARGET_LAYERS = ['.*']

# 客製化顏色定義
LAYER_COLORS = {
    # METAL Layers
    'METAL0': '#4169E1', 'M0': '#4169E1', # RoyalBlue
    'METAL1': '#DC143C', 'M1': '#DC143C', # Crimson
    'METAL2': '#32CD32', 'M2': '#32CD32', # LimeGreen
    'METAL3': '#FFD700', 'M3': '#FFD700', # Gold
    'METAL4': '#8A2BE2', 'M4': '#8A2BE2', # BlueViolet
    'METAL5': '#CD853F', 'M5': '#CD853F', # Peru
    'METAL6': '#FF00FF', 'M6': '#FF00FF', # Magenta
    'METAL7': '#00CED1', 'M7': '#00CED1', # DarkTurquoise
    'METAL8': '#8B4513', 'M8': '#8B4513', # SaddleBrown
    'METAL9': '#FFFF00', 'M9': '#FFFF00', # Yellow
    'METAL10': '#9ACD32', 'M10': '#9ACD32', # YellowGreen
    'METAL11': '#FF1493', 'M11': '#FF1493', # DeepPink
    'METAL12': '#DEB887', 'M12': '#DEB887', # BurlyWood
    'METAL13': '#008B8B', 'M13': '#008B8B', # DarkCyan
    'METAL14': '#9370DB', 'M14': '#9370DB', # MediumPurple
    'METAL15': '#556B2F', 'M15': '#556B2F', # DarkOliveGreen

    # VIA Layers
    'VIA0': '#FF4500', 'V0': '#FF4500', # OrangeRed
    'VIA1': '#32CD32', 'V1': '#32CD32', # LimeGreen
    'VIA2': '#ADFF2F', 'V2': '#ADFF2F', # GreenYellow
    'VIA3': '#D2691E', 'V3': '#D2691E', # Chocolate
    'VIA4': '#F4A460', 'V4': '#F4A460', # SandyBrown
    'VIA5': '#DA70D6', 'V5': '#DA70D6', # Orchid
    'VIA6': '#40E0D0', 'V6': '#40E0D0', # Turquoise
    'VIA7': '#A0522D', 'V7': '#A0522D', # Sienna
    'VIA8': '#F0E68C', 'V8': '#F0E68C', # Khaki
    'VIA9': '#00FF00', 'V9': '#00FF00', # Lime
    'VIA10': '#FF69B4', 'V10': '#FF69B4', # HotPink
    'VIA11': '#CD853F', 'V11': '#CD853F', # Peru
    'VIA12': '#00FFFF', 'V12': '#00FFFF', # Cyan
    'VIA13': '#708090', 'V13': '#708090', # SlateGray
    'VIA14': '#FF0000', 'V14': '#FF0000', # Red
    
    # Fallback
    'VIA': '#FFFF00', 
}

def get_layer_style(layer_name):
    """
    根據圖層名稱設定顏色和填充樣式。
    """
    name = layer_name.upper()
    
    # 特殊層處理
    if 'OBS' in name:
        return ('#808080', '++') # OBS 使用 '++' (十字)
        
    # 輔助函式：根據 Metal Index 決定紋理
    def get_metal_hatch(idx):
        # 偶數層 /// (右斜), 奇數層 \\\ (左斜)
        if idx % 2 == 0:
            return '///'
        else:
            return '\\\\\\' 

    # 1. 嘗試直接匹配名稱 (移除括號)
    clean_name = re.sub(r'\(.*?\)', '', name).strip()
    
    if clean_name in LAYER_COLORS:
        color = LAYER_COLORS[clean_name]
        
        # 判斷是否為 VIA
        if 'VIA' in clean_name or name.startswith('V'):
            hatch = '***' # VIA 使用 '***'
        else:
            # 是 Metal，嘗試提取數字來決定紋理方向
            match = re.search(r'(?:METAL|M)(\d+)', clean_name)
            if match:
                idx = int(match.group(1))
                hatch = get_metal_hatch(idx)
            else:
                hatch = '///' # 預設 Metal 紋理
        return (color, hatch)

    # 2. 嘗試解析數字 (模糊匹配)
    metal_match = re.search(r'(?:METAL|M)(\d+)', name)
    if metal_match:
        idx = int(metal_match.group(1))
        std_key = f'METAL{idx}'
        if std_key in LAYER_COLORS:
            # Metal 紋理交錯
            hatch = get_metal_hatch(idx)
            return (LAYER_COLORS[std_key], hatch)
    
    via_match = re.search(r'(?:VIA|V)(\d+)', name)
    if via_match:
        idx = int(via_match.group(1))
        std_key = f'VIA{idx}'
        if std_key in LAYER_COLORS:
            return (LAYER_COLORS[std_key], '***') # VIA 使用 '***'

    # 3. 通用 VIA 匹配 (如果沒有數字)
    if 'VIA' in name or name.startswith('V'):
        return (LAYER_COLORS.get('VIA', '#FFFF00'), '***')

    # 4. 未知層 (Masterslice 等)
    return ('#FF00FF', '...') # 紫紅色點狀

def plot_macros(macros, output_file_base, db_microns=1.0, target_cells=None, target_layers=None, theme='dark', show_pins=True, show_obs=True):
    """
    繪製 Macro 的幾何形狀 (Pin, OBS)，支援亮暗主題切換及圖層過濾。
    """
    
    # 使用預設圖層過濾器如果使用者沒有提供
    current_target_layers = target_layers if target_layers else DEFAULT_TARGET_LAYERS

    # --- 內部函式：判斷圖層是否顯示 ---
    def is_layer_visible(layer_name):
        # 依據 current_target_layers 進行正規表達式匹配
        for pattern in current_target_layers:
            try:
                # 使用 re.search 進行模糊匹配 (例如 'M1' 可以匹配 'METAL1')
                if re.search(pattern, layer_name, re.IGNORECASE):
                    return True
            except re.error:
                print(f"[警告] 無效的正規表達式: {pattern}")
        return False

    # --- 過濾邏輯 (Cell) ---
    if target_cells:
        target_set = set(target_cells)
        macros_to_plot = [m for m in macros if m['name'] in target_set]
        
        found_names = set(m['name'] for m in macros_to_plot)
        missing = target_set - found_names
        if missing:
            print(f"[繪圖提示] 以下指定的 Cell 在 JSON 中未找到: {', '.join(missing)}")
    else:
        # 預設行為：過濾掉沒有尺寸的 Macro，取前 5 個
        macros_to_plot = [m for m in macros if m.get('size', {}).get('width', 0) > 0][:5]
    
    if not macros_to_plot:
        print("[繪圖] 沒有符合條件的 Macro 資料可供繪製。")
        return

    layer_msg = f"指定圖層: {', '.join(current_target_layers)}"
    display_opts = []
    if show_pins: display_opts.append("Pins")
    if show_obs: display_opts.append("OBS")
    
    print(f"\n[繪圖] 正在繪製 {len(macros_to_plot)} 個 Macro ({theme} 主題, {layer_msg}, 顯示: {', '.join(display_opts)})...")

    # 設定主題顏色變數
    if theme == 'dark':
        plt.style.use('dark_background')
        ui_color = 'white'       # 文字、邊框顏色
        grid_color = '#404040'   # 格線顏色
        legend_bg = '#202020'    # 圖例背景
        boundary_color = 'white' # Macro 邊界
        current_pin_colors = PIN_DIRECTION_COLORS_DARK
    else:
        plt.style.use('default') # 預設淺色主題
        ui_color = 'black'
        grid_color = '#cccccc'
        legend_bg = '#f5f5f5'
        boundary_color = 'black'
        current_pin_colors = PIN_DIRECTION_COLORS_LIGHT

    # 動態調整畫布高度
    fig_height = 6 + (len(macros_to_plot) * 3) 
    if fig_height > 50: fig_height = 50 
    
    fig, ax = plt.subplots(figsize=(20, 12)) 
    ax.set_aspect('equal') 
    
    current_x_offset = 0.0
    max_height = 0.0
    
    legend_elements = {}

    for macro in macros_to_plot:
        width = macro['size']['width']
        height = macro['size']['height']
        
        origin_x = current_x_offset
        origin_y = 0.0 
        
        # --- 1. 繪製 Macro 邊界 (Boundary) ---
        ax.add_patch(patches.Rectangle(
            (origin_x, origin_y), width, height,
            edgecolor=boundary_color, facecolor='none', 
            linestyle='--', linewidth=1.5, alpha=0.8,
            zorder=10
        ))
        
        # 標註 Macro 名稱 (寫在 Macro 邊界內的左下角)
        # 加入微小偏移 (2%) 以避免壓線
        text_x = origin_x + (width * 0.02)
        text_y = origin_y + (height * 0.02)
        
        ax.text(text_x, text_y, macro['name'], 
                color=ui_color, fontsize=14, fontweight='bold', 
                ha='left', va='bottom', zorder=20) 

        # --- 2. 繪製 OBS (Obstructions) ---
        if show_obs:
            for obs_layer in macro.get('obs', []):
                layer_name = obs_layer['layer']
                
                # [更新] 圖層過濾檢查
                if not is_layer_visible(layer_name):
                    continue

                base_color, hatch_style = get_layer_style(layer_name) 
                
                # OBS 統一使用 '++' (密度2十字)
                real_hatch = '++' 
                
                for rect in obs_layer['rects']:
                    x = origin_x + rect['llx']
                    y = origin_y + rect['lly']
                    w = rect['urx'] - rect['llx']
                    h = rect['ury'] - rect['lly']
                    
                    # OBS 使用十字紋理，邊框使用層顏色
                    obs_patch = patches.Rectangle(
                        (x, y), w, h,
                        edgecolor=base_color, 
                        facecolor='none',     
                        hatch=real_hatch, 
                        linewidth=0.5,
                        alpha=0.6,
                        zorder=2
                    )
                    ax.add_patch(obs_patch)
                    
                    # [修正] 確保 OBS 與對應圖層都在 Legend 中
                    if 'OBS' not in legend_elements:
                        legend_elements['OBS'] = patches.Patch(facecolor='none', edgecolor='gray', hatch=real_hatch, label='OBS (Obstruction)')
                    
                    # 即使該層只有 OBS，也要在圖例中顯示該層的顏色對照
                    if layer_name not in legend_elements:
                        legend_elements[layer_name] = patches.Patch(facecolor='none', edgecolor=base_color, hatch=hatch_style, label=layer_name)

        # --- 3. 繪製 PINs ---
        if show_pins:
            for pin in macro['pins']:
                pin_name = pin['name']
                pin_direction = pin.get('direction', 'UNKNOWN').upper()
                # 取得 Pin Direction 顏色作為文字顏色 (使用當前主題的顏色表)
                text_color = current_pin_colors.get(pin_direction, ui_color)

                for rect_info in pin['rects']:
                    layer_name = rect_info['layer']
                    
                    # [更新] 圖層過濾檢查
                    if not is_layer_visible(layer_name):
                        continue

                    # 獲取層樣式 (包含交錯紋理)
                    color, default_hatch = get_layer_style(layer_name)
                    
                    x = origin_x + rect_info['llx']
                    y = origin_y + rect_info['lly']
                    w = rect_info['urx'] - rect_info['llx']
                    h = rect_info['ury'] - rect_info['lly']
                    
                    # 顏色處理 (HEX -> RGBA with alpha)
                    edge_c = color
                    face_c = color + '40' # Hex alpha (25% opacity)
                    
                    rect_patch = patches.Rectangle(
                        (x, y), w, h,
                        linewidth=1,
                        edgecolor=edge_c,
                        facecolor=face_c, 
                        hatch=default_hatch * 2, # 加密紋理密度
                        zorder=5
                    )
                    ax.add_patch(rect_patch)
                    
                    # 收集圖例 (Layer) - facecolor='none'
                    if layer_name not in legend_elements:
                        legend_elements[layer_name] = patches.Patch(facecolor='none', edgecolor=edge_c, hatch=default_hatch * 2, label=layer_name)
                    
                    # --- 4. Pin 名稱標註 ---
                    center_x = x + w / 2
                    center_y = y + h / 2
                    
                    # 標註文字顏色使用 Pin Direction，移除 bbox 背景，字體放大
                    ax.text(center_x, center_y, pin_name,
                            color=text_color, fontsize=14, fontweight='bold',
                            ha='center', va='center', zorder=20)

        current_x_offset += width * 1.2
        max_height = max(max_height, height)

    # --- 5. 圖片置中與裝飾 ---
    
    padding_x = current_x_offset * 0.05 if current_x_offset > 0 else 1.0
    padding_y = max_height * 0.1 if max_height > 0 else 1.0
    ax.set_xlim(-padding_x, current_x_offset)
    ax.set_ylim(-padding_y, max_height + padding_y)
    
    # [更新] 計算比例與尺寸資訊
    scale_val = 1.0 / db_microns if db_microns else 1.0
    scale_info = f"1 LEF Unit = {scale_val} µm" if db_microns else "Units: Unknown"
    
    # [更新] 標題包含 Cell 尺寸
    cell_details = [f"{m['name']}({m['size']['width']*scale_val:.3f}x{m['size']['height']*scale_val:.3f})" for m in macros_to_plot]
    cell_str = ", ".join(cell_details)
    
    if len(cell_str) > 80:
        cell_info = f"Cells: {cell_str[:77]}..."
    else:
        cell_info = f"Cells: {cell_str}"

    ax.set_title(f"LEF Layout View\n{scale_info} | {cell_info}", color=ui_color, fontsize=16, pad=20)
    ax.set_xlabel("X (microns)", color=ui_color)
    ax.set_ylabel("Y (microns)", color=ui_color)
    
    ax.grid(True, which='both', color=grid_color, linestyle=':', linewidth=0.5)
    for spine in ax.spines.values():
        spine.set_edgecolor(grid_color)
    ax.tick_params(axis='x', colors=ui_color)
    ax.tick_params(axis='y', colors=ui_color)

    # 圖例排序
    def sort_key(key):
        if 'OBS' in key: return 9000
        
        num_match = re.search(r'\d+', key)
        num = int(num_match.group()) if num_match else -1
        
        base_score = num * 10
        if 'VIA' in key or key.startswith('V'):
            base_score += 1
        return base_score
        
    sorted_keys = sorted(legend_elements.keys(), key=sort_key)
    handles = [legend_elements[k] for k in sorted_keys]
    
    # [更新] 確保 Pin Direction 顏色也在圖例中，並且填充顏色 (使用當前主題顏色)
    direction_handles = [
        patches.Patch(facecolor=current_pin_colors['INPUT'], edgecolor=current_pin_colors['INPUT'], alpha=0.7, label='INPUT (Text)'),
        patches.Patch(facecolor=current_pin_colors['OUTPUT'], edgecolor=current_pin_colors['OUTPUT'], alpha=0.7, label='OUTPUT (Text)'),
        patches.Patch(facecolor=current_pin_colors['INOUT'], edgecolor=current_pin_colors['INOUT'], alpha=0.7, label='INOUT (Text)'),
    ]
    
    all_handles = handles + direction_handles

    if all_handles:
        # 圖例顯示在右方，單列顯示
        lgd = ax.legend(handles=all_handles, loc='upper left', bbox_to_anchor=(1.05, 1.0),
                  ncol=1, facecolor=legend_bg, edgecolor=ui_color, labelcolor=ui_color)

    plot_filename = f"{output_file_base}_layout.png"
    
    # [更新] 使用 bbox_inches='tight' 替代 tight_layout()
    plt.savefig(plot_filename, dpi=150, facecolor=fig.get_facecolor(), bbox_extra_artists=(lgd,) if all_handles else None, bbox_inches='tight')
    print(f"\n[繪圖] 圖片已儲存至: {plot_filename}")
    plt.close(fig)

# --- 主程式 ---

def main():
    parser = argparse.ArgumentParser(description='LEF Visualizer (Reads Parsed JSON)')
    parser.add_argument('json_file', help='Path to the parsed JSON file (output from lef_parser.py)')
    parser.add_argument('--cells', nargs='+', help='List of cell names to plot, or a file containing the list')
    parser.add_argument('--layers', nargs='+', help='List of regex patterns for layers to show (e.g., METAL1, M.*, VIA[1-3])')
    parser.add_argument('--theme', choices=['dark', 'light'], default='dark', help='Color theme for visualization')
    
    # 新增：控制 PIN 和 OBS 顯示的選項
    parser.add_argument('--hide_pins', action='store_true', help='Hide PIN geometries')
    parser.add_argument('--hide_obs', action='store_true', help='Hide OBS geometries')
    
    args = parser.parse_args()
    
    # 檢查 JSON 檔案是否存在
    if not os.path.exists(args.json_file):
        print(f"[錯誤] 找不到 JSON 檔案: {args.json_file}")
        sys.exit(1)

    # 讀取 JSON 資料
    print(f"正在讀取資料: {args.json_file} ...")
    try:
        with open(args.json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"[錯誤] 無法讀取或解析 JSON 檔案: {e}")
        sys.exit(1)

    macros = data.get('macros', [])
    db_microns = data.get('database_microns', 1000.0)
    
    if not macros:
        print("[警告] JSON 檔案中沒有 Macro 資料。")
        sys.exit(0)

    # 處理 Cell List
    target_cells = []
    if args.cells:
        if len(args.cells) == 1 and os.path.isfile(args.cells[0]):
            try:
                with open(args.cells[0], 'r') as f:
                    target_cells = [line.strip() for line in f if line.strip()]
                print(f"[設定] 從檔案載入 {len(target_cells)} 個目標 Cell。")
            except Exception as e:
                print(f"[錯誤] 無法讀取 Cell 清單檔案: {e}")
                sys.exit(1)
        else:
            target_cells = args.cells
            print(f"[設定] 指定顯示 {len(target_cells)} 個 Cell: {', '.join(target_cells)}")

    # 執行繪圖
    output_base = os.path.splitext(args.json_file)[0]
    plot_macros(
        macros, 
        output_base, 
        db_microns, 
        target_cells=target_cells, 
        target_layers=args.layers, 
        theme=args.theme,
        show_pins=not args.hide_pins, # 預設顯示 (True)，若有 flag 則為 False
        show_obs=not args.hide_obs    # 預設顯示 (True)，若有 flag 則為 False
    )

if __name__ == "__main__":
    main()