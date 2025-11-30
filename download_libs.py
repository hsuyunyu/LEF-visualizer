import os
import urllib.request
import ssl

# 設定下載目錄
LIB_DIR = "lib"

# 依賴套件清單 (對應 dashboard_offline.html 中的 CDN 連結)
LIBRARIES = {
    "react.production.min.js": "https://unpkg.com/react@18/umd/react.production.min.js",
    "react-dom.production.min.js": "https://unpkg.com/react-dom@18/umd/react-dom.production.min.js",
    "babel.min.js": "https://unpkg.com/@babel/standalone/babel.min.js",
    "tailwindcss.js": "https://cdn.tailwindcss.com",
    "lucide.min.js": "https://unpkg.com/lucide@latest/dist/umd/lucide.min.js"
}

def download_libs():
    # 建立資料夾
    if not os.path.exists(LIB_DIR):
        os.makedirs(LIB_DIR)
        print(f"已建立目錄: {LIB_DIR}")

    # 忽略 SSL 憑證驗證
    ssl._create_default_https_context = ssl._create_unverified_context

    print("開始下載依賴檔案...")
    
    for filename, url in LIBRARIES.items():
        filepath = os.path.join(LIB_DIR, filename)
        print(f"正在下載: {filename} ...")
        try:
            urllib.request.urlretrieve(url, filepath)
            print(f"  -> 完成: {filepath}")
        except Exception as e:
            print(f"  -> 失敗: {url}\n     錯誤: {e}")

    print("\n下載完成！")
    print("請將 'lib' 資料夾與 'dashboard_offline.html' 一起複製到離線電腦上。")

if __name__ == "__main__":
    download_libs()