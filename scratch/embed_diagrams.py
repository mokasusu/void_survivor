import os

filepath = "/Users/vducc3110/Desktop/void_survivor/chuong4.md"

if not os.path.exists(filepath):
    print(f"Error: {filepath} does not exist.")
    exit(1)

with open(filepath, 'r', encoding='utf-8') as f:
    content = f.read()

# Replace block 1 (Model Zip Structure)
block_1_old = """> [!NOTE]
> **[ẢNH CẦN CHÈN 1 - CẤU TRÚC FILE ZIP MODEL]**
> *   **Vị trí chèn:** Ngay sau phần phân tích cấu trúc file `.zip`.
> *   **Tên file đề xuất:** `model_zip_structure.png` (Sơ đồ mô tả cấu trúc thư mục giải nén của file model `.zip` gồm 4 thành phần trên để tăng tính trực quan khoa học).
> *   **Chú thích:** *Hình 4.9: Sơ đồ mô tả cấu trúc tuần tự hóa các tệp tin lưu trữ bên trong định dạng checkpoint .zip của Stable-Baselines3.*"""

block_1_new = """![Hình 4.9: Sơ đồ mô tả cấu trúc tuần tự hóa các tệp tin lưu trữ bên trong định dạng checkpoint .zip của Stable-Baselines3](model_zip_structure.png)
*Hình 4.9: Sơ đồ mô tả cấu trúc tuần tự hóa các tệp tin lưu trữ bên trong định dạng checkpoint .zip của Stable-Baselines3.*"""

# Replace block 2 (Inference Flowchart)
block_2_old = """> [!NOTE]
> **[ẢNH CẦN CHÈN 2 - SƠ ĐỒ CHU TRÌNH INFERENCE]**
> *   **Vị trí chèn:** Ngay sau các kỹ thuật tối ưu hóa của quá trình suy luận.
> *   **Tên file đề xuất:** `inference_flowchart.png` (Sơ đồ khối thể hiện vòng lặp suy luận: Game State -> PPOStateEncoder -> Action Masking -> model.predict(deterministic=True) -> Action execution -> pygame.display.flip()).
> *   **Chú thích:** *Hình 4.10: Sơ đồ luồng xử lý và vòng lặp suy luận thời gian thực của Agent trong kịch bản chạy trình diễn (enjoy.py).*"""

block_2_new = """![Hình 4.10: Sơ đồ luồng xử lý và vòng lặp suy luận thời gian thực của Agent trong kịch bản chạy trình diễn (enjoy.py)](inference_flowchart.png)
*Hình 4.10: Sơ đồ luồng xử lý và vòng lặp suy luận thời gian thực của Agent trong kịch bản chạy trình diễn (enjoy.py).*"""

# Replace block 3 (Game Interface Layout)
block_3_old = """> [!NOTE]
> **[ẢNH CẦN CHÈN 3 - SƠ ĐỒ THIẾT KẾ GIAO DIỆN GAME]**
> *   **Vị trí chèn:** Ngay sau phần mô tả các thông số HUD và bảng điều khiển.
> *   **Tên file đề xuất:** `game_interface_layout.png` (Sơ đồ phác họa thiết kế màn hình game 800x600, chỉ rõ vị trí thanh Boss HP, Player HP, Info Panel cao 90px và Hitbox 15px/Grazing zone 50px của phi thuyền).
> *   **Chú thích:** *Hình 4.11: Bản phác thảo sơ đồ bố trí các thực thể và khu vực thông số hiển thị (HUD) trên giao diện game.*"""

block_3_new = """![Hình 4.11: Bản phác thảo sơ đồ bố trí các thực thể và khu vực thông số hiển thị (HUD) trên giao diện game](game_interface_layout.png)
*Hình 4.11: Bản phác thảo sơ đồ bố trí các thực thể và khu vực thông số hiển thị (HUD) trên giao diện game.*"""

# Apply replacements
replacements = [
    (block_1_old, block_1_new),
    (block_2_old, block_2_new),
    (block_3_old, block_3_new)
]

replaced_count = 0
for old, new in replacements:
    if old in content:
        content = content.replace(old, new)
        replaced_count += 1
    else:
        print(f"Warning: Block not found in chuong4.md.")

with open(filepath, 'w', encoding='utf-8') as f:
    f.write(content)

print(f"Successfully embedded {replaced_count}/3 diagrams in chuong4.md.")
