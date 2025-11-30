import sys
import json
import os
import argparse
import glob
from multiprocessing import Pool, cpu_count
import time
import subprocess

# --- 核心單檔解析邏輯 (Functional Style) ---

def parse_lef_content(filepath):
    """
    解析單一 LEF 檔案，回傳該檔案中發現的所有 Macro 列表，以及屬性定義和資料庫單位。
    支援解析：
    - MACRO 基礎資訊 (Size, Origin)
    - PIN 詳細資訊 (Direction, Use, Layer, Rect/Mask, Antenna)
    - OBS (障礙物) 資訊 (Layer, Rect/Mask)
    """
    print(f"   -> [PID {os.getpid()}] 正在解析: {filepath}")
    
    macros_dict = {} 
    property_definitions = {} 
    
    # 狀態變數
    current_macro = None
    current_pin = None
    current_layer = None
    
    # 區塊旗標
    in_obs_block = False          
    current_obs_layer_name = None 
    
    # [修正] 變數名稱修正，使其與下方迴圈邏輯一致
    is_parsing_property_definitions = False      
    
    db_microns = 1000.0 # 預設值
    
    if not os.path.exists(filepath):
        print(f"      [警告] 找不到檔案: {filepath}，已略過。")
        return [], {}, db_microns

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except Exception:
        # Fallback encoding
        try:
            with open(filepath, 'r', encoding='latin-1') as f:
                lines = f.readlines()
        except Exception as e:
            print(f"      [錯誤] 無法讀取檔案 {filepath}: {e}")
            return [], {}, db_microns

    # 內部輔助函式：結束當前 PIN 處理
    def finalize_pin():
        nonlocal current_pin, current_layer
        if current_pin and current_macro:
            if current_pin['rects']: 
                current_macro['pins'].append(current_pin)
            current_pin = None
            current_layer = None

    # 內部輔助函式：結束當前 MACRO 處理
    def finalize_macro():
        nonlocal current_macro, in_obs_block
        finalize_pin() 
        if current_macro:
            macros_dict[current_macro['name']] = current_macro 
            current_macro = None
            in_obs_block = False 

    source_file_name = os.path.basename(filepath)

    for line in lines:
        line = line.split('#')[0].strip()
        if not line: continue
        parts = line.split()
        keyword = parts[0].upper()

        # --- 1. 資料庫單位與屬性定義 ---
        if keyword == 'DATABASE':
            if len(parts) >= 3 and parts[1].upper() == 'MICRONS':
                try:
                    db_microns = float(parts[2].rstrip(';'))
                except ValueError: pass
            continue

        if keyword == 'PROPERTYDEFINITIONS':
            is_parsing_property_definitions = True
            continue
        
        if keyword == 'END' and len(parts) > 1 and parts[1] == 'PROPERTYDEFINITIONS':
            is_parsing_property_definitions = False
            continue

        if is_parsing_property_definitions:
            if len(parts) >= 3 and parts[-1].endswith(';'):
                prop_name = parts[1]
                # 簡單儲存屬性定義，不進行深度解析
                property_definitions[prop_name] = line.strip()
            continue 

        # --- 2. MACRO 區塊 ---
        if keyword == 'MACRO':
            finalize_macro()
            current_macro = {
                "name": parts[1],
                "source_file": source_file_name,
                "size": {"width": 0.0, "height": 0.0},
                "origin": {"x": 0.0, "y": 0.0},
                "pins": [],
                "obs": [] 
            }
        
        elif keyword == 'END' and len(parts) > 1 and current_macro and parts[1] == current_macro['name']:
            finalize_macro()

        # --- 3. MACRO 內部解析 ---
        elif current_macro:
            
            # --- 3a. PIN 定義 ---
            if keyword == 'PIN':
                finalize_pin()
                current_pin = {
                    "name": parts[1],
                    "direction": "UNKNOWN",
                    "use": "SIGNAL",
                    "rects": [],
                    "antenna": {} 
                }
            
            elif current_pin:
                if keyword == 'END' and len(parts) > 1 and parts[1] == current_pin['name']:
                    finalize_pin()
                
                elif keyword == 'DIRECTION':
                    current_pin['direction'] = parts[1].rstrip(';').upper()
                elif keyword == 'USE':
                    current_pin['use'] = parts[1].rstrip(';').upper()
                elif keyword.startswith('ANTENNA'):
                    prop_name = keyword.lower()
                    try:
                        val = float(parts[1].rstrip(';'))
                        current_pin['antenna'][prop_name] = val
                    except ValueError: pass
                elif keyword == 'LAYER':
                    current_layer = parts[1].rstrip(';')
                elif keyword == 'RECT':
                    if current_layer:
                        try:
                            # 處理 RECT (含 MASK)
                            coords_start_idx = 1
                            mask_val = None
                            if parts[1].upper() == 'MASK':
                                mask_val = int(parts[2])
                                coords_start_idx = 3
                            
                            rect = {
                                "layer": current_layer,
                                "mask": mask_val,
                                "llx": float(parts[coords_start_idx]),
                                "lly": float(parts[coords_start_idx+1]),
                                "urx": float(parts[coords_start_idx+2]),
                                "ury": float(parts[coords_start_idx+3].rstrip(';'))
                            }
                            current_pin['rects'].append(rect)
                        except (IndexError, ValueError): pass

            # --- 3b. OBS (Obstruction) 定義 ---
            elif keyword == 'OBS':
                in_obs_block = True
                current_obs_layer_name = None
            
            elif in_obs_block:
                if keyword == 'END': 
                    in_obs_block = False
                    current_obs_layer_name = None
                elif keyword == 'LAYER':
                    current_obs_layer_name = parts[1].rstrip(';')
                    # 如果該 Layer 還沒有在 OBS 列表中，則新增
                    if not any(o['layer'] == current_obs_layer_name for o in current_macro['obs']):
                        current_macro['obs'].append({'layer': current_obs_layer_name, 'rects': []})
                elif keyword == 'RECT' and current_obs_layer_name:
                    try:
                        # 處理 OBS RECT (含 MASK)
                        coords_start_idx = 1
                        mask_val = None
                        if parts[1].upper() == 'MASK':
                            mask_val = int(parts[2])
                            coords_start_idx = 3

                        rect = {
                            "llx": float(parts[coords_start_idx]),
                            "lly": float(parts[coords_start_idx+1]),
                            "urx": float(parts[coords_start_idx+2]),
                            "ury": float(parts[coords_start_idx+3].rstrip(';')),
                            "mask": mask_val
                        }
                        # 找到對應的 Layer 並加入 RECT
                        for obs_entry in current_macro['obs']:
                            if obs_entry['layer'] == current_obs_layer_name:
                                obs_entry['rects'].append(rect)
                                break
                    except (IndexError, ValueError): pass

            # --- 3c. 一般屬性 (SIZE, ORIGIN) ---
            elif not in_obs_block and not current_pin:
                if keyword == 'SIZE':
                    try:
                        current_macro['size'] = {
                            "width": float(parts[1]),
                            "height": float(parts[3].rstrip(';'))
                        }
                    except (IndexError, ValueError): pass
                elif keyword == 'ORIGIN':
                    try:
                        current_macro['origin'] = {
                            "x": float(parts[1]),
                            "y": float(parts[2].rstrip(';'))
                        }
                    except (IndexError, ValueError): pass

    finalize_macro()
    
    print(f"      -> [PID {os.getpid()}] 找到 {len(macros_dict)} 個不重複 Macro，處理完成。")
    return list(macros_dict.values()), property_definitions, db_microns

# --- 檔案列表處理 ---

def get_file_list(inputs):
    """處理輸入參數，展開 wildcards 或讀取 list 檔案"""
    files_to_process = []
    
    for input_arg in inputs:
        if input_arg.endswith('.list') or input_arg.endswith('.txt'):
            try:
                base_dir = os.path.dirname(os.path.abspath(input_arg))
                with open(input_arg, 'r', encoding='utf-8') as f:
                    for line in f:
                        path = line.strip()
                        if path and not path.startswith('#'):
                            # Handle relative paths
                            full_path = path if os.path.isabs(path) else os.path.join(base_dir, path)
                            files_to_process.append(full_path)
                print(f"[INFO] Paths loaded from list file '{input_arg}'.")
            except Exception as e:
                print(f"[ERROR] Could not read list file {input_arg}: {e}")
        
        else:
            expanded = glob.glob(input_arg)
            if not expanded:
                files_to_process.append(input_arg)
            else:
                files_to_process.extend(expanded)
                
    seen = set()
    unique_files = []
    for f in files_to_process:
        if f not in seen and os.path.exists(f):
            unique_files.append(f)
            seen.add(f)
            
    return unique_files

# --- Main Controller Function (using multiprocessing) ---

def main():
    parser = argparse.ArgumentParser(description='LEF Multi-File Parser (Parallel Processing) with Visualization')
    parser.add_argument('inputs', nargs='+', 
                        help='Input file paths. Can be multiple .lef files or a .list/.txt file containing paths.')
    parser.add_argument('-o', '--output', default='lef_data.json',
                        help='Output JSON file path (default: lef_data.json)')
    parser.add_argument('-j', '--jobs', type=int, default=cpu_count(),
                        help=f'Number of parallel processes (default: CPU count: {cpu_count()})')
    parser.add_argument('--plot', action='store_true',
                        help='Enables Macro visualization, which outputs a PNG image (requires matplotlib and numpy).')
    
    # 視覺化工具相關參數
    parser.add_argument('--cells', nargs='+', help='List of cell names to plot (passed to visualizer)')
    parser.add_argument('--layers', nargs='+', help='Regex patterns for layers to show (passed to visualizer)')
    parser.add_argument('--theme', choices=['dark', 'light'], default='dark', help='Color theme (passed to visualizer)')

    args = parser.parse_args()
    
    # 1. Get all files to be processed
    target_files = get_file_list(args.inputs)
    
    if not target_files:
        print("[ERROR] No valid input files found.")
        sys.exit(1)
    
    num_workers = min(args.jobs, len(target_files))
    
    print(f"=== Starting Batch Parsing ({len(target_files)} files) ===")
    print(f"Using {num_workers} parallel processes")
    
    start_time = time.time()
    
    # 2. Start parallel processing Pool
    all_macros_dict = {} 
    all_property_definitions = {} 
    global_db_microns = 1.0 # Global scale factor for plotting (default 1.0)
    
    try:
        with Pool(processes=num_workers) as pool:
            # pool.map now returns a list of tuples: List[Tuple[List[Macro], Dict[str, Any], float]]
            results = pool.map(parse_lef_content, target_files)

        # 3. Collect and flatten the results
        for macro_list, props_dict, db_microns_file in results: # Unpack the tuple
            # Collect Macros (as before)
            for macro in macro_list:
                all_macros_dict[macro['name']] = macro
            
            # Collect Property Definitions
            # Simple merge: last file parsed will overwrite definitions (Last definition wins).
            all_property_definitions.update(props_dict) 
            
            # Collect Database Microns (take the last one found, assuming consistent files)
            if db_microns_file != 1.0:
                global_db_microns = db_microns_file
            
    except Exception as e:
        print(f"\n[FATAL ERROR] Parallel processing interrupted: {e}")
        sys.exit(1)
        
    end_time = time.time()
    
    # Convert dictionary values back to a list for the final output
    all_macros = list(all_macros_dict.values())
    
    # 4. Output JSON results
    final_data = {
        "database_microns": global_db_microns, # Include the scale factor in output
        "macros": all_macros,
        "property_definitions": all_property_definitions, 
        "total_macros": len(all_macros),
        "total_property_definitions": len(all_property_definitions), 
        "source_files_count": len(target_files)
    }
    
    try:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(final_data, f, indent=2)
            
        print(f"\n=== Success! ===")
        print(f"Total files parsed: {len(target_files)}")
        print(f"Total unique Macros extracted: {len(all_macros)}")
        print(f"Total property definitions extracted: {len(all_property_definitions)}")
        print(f"Using scale factor (Database Microns): {global_db_microns}")
        print(f"Total time elapsed: {end_time - start_time:.2f} seconds")
        print(f"Results saved to: {args.output}")
    except Exception as e:
        print(f"\n[ERROR] Could not write output file: {e}")

    # 5. Call external visualizer
    if args.plot:
        visualizer_script = 'lef_visualizer.py'
        
        if not os.path.exists(visualizer_script):
            script_dir = os.path.dirname(os.path.abspath(__file__))
            visualizer_script = os.path.join(script_dir, 'lef_visualizer.py')

        if os.path.exists(visualizer_script):
            print(f"\n[Parser] 正在呼叫視覺化工具: {visualizer_script} ...")
            
            cmd = [sys.executable, visualizer_script, args.output]
            
            if args.cells:
                cmd.append('--cells')
                cmd.extend(args.cells)
            
            if args.layers:
                cmd.append('--layers')
                cmd.extend(args.layers)
                
            if args.theme:
                cmd.extend(['--theme', args.theme])
            
            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as e:
                print(f"[錯誤] 視覺化工具執行失敗: {e}")
        else:
            print(f"\n[警告] 找不到 '{visualizer_script}'，無法執行繪圖。請確認該檔案存在。")


if __name__ == "__main__":
    main()