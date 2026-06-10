import os

filepath = "/Users/vducc3110/Desktop/void_survivor/chuong4.md"
with open(filepath, 'r', encoding='utf-8') as f:
    content = f.read()

idx = content.find("## 4.3. Đóng gói mô hình")
if idx != -1:
    print("Found ## 4.3. text:")
    print(content[idx:idx+1500]) # Print first 1500 characters
else:
    print("## 4.3. Đóng gói mô hình not found.")
