import os

src_file = "/Users/vducc3110/Desktop/void_survivor/chuong4_phan4.1.md"
dest_file = "/Users/vducc3110/Desktop/void_survivor/chuong4.md"

if not os.path.exists(src_file):
    print(f"Error: {src_file} does not exist.")
    exit(1)

with open(src_file, 'r', encoding='utf-8') as f:
    content = f.read()

# Define the placeholders and their replacements
replacements = {
    # 1. Win Rate
    """> [!IMPORTANT]
> **[CHÈN ẢNH ĐỒ THỊ TỶ LỆ THẮNG TẠI ĐÂY]**
> *   **Tên ảnh:** `evaluation_win_rate.png`
> *   **Chú thích dưới hình:** *Hình 4.1: Biểu đồ tỷ lệ thắng (Win Rate %) của Agent trên các Stage huấn luyện khác nhau, thể hiện độ ổn định vượt ngưỡng 80% trước khi thăng Stage.*""": 
    """![Hình 4.1: Biểu đồ tỷ lệ thắng (Win Rate %)](evaluation_win_rate.png)
*Hình 4.1: Biểu đồ tỷ lệ thắng (Win Rate %) của Agent trên các Stage huấn luyện khác nhau, thể hiện độ ổn định vượt ngưỡng 80% trước khi thăng Stage.*""",

    # 2. Reward
    """> [!IMPORTANT]
> **[CHÈN ẢNH BIỂU ĐỒ REWARD TẠI ĐÂY]**
> *   **Tên ảnh:** `rollout_ep_rew_mean.png` (Biểu đồ thứ 2 người dùng tải lên).
> *   **Chú thích dưới hình:** *Hình 4.2: Biểu đồ phần thưởng trung bình (rollout/ep_rew_mean) ghi nhận từ Tensorboard, thể hiện giá trị tăng từ mốc ~140 lên 202.49 và vết võng U-shape tại step 12M.*""":
    """![Hình 4.2: Biểu đồ phần thưởng trung bình (rollout/ep_rew_mean)](rollout_ep_rew_mean.png)
*Hình 4.2: Biểu đồ phần thưởng trung bình (rollout/ep_rew_mean) ghi nhận từ Tensorboard, thể hiện giá trị tăng từ mốc ~140 lên 202.49 và vết võng U-shape tại step 12M.*""",

    # 3. Episode Length
    """> [!IMPORTANT]
> **[CHÈN ẢNH BIỂU ĐỒ EPISODE LENGTH TẠI ĐÂY]**
> *   **Tên ảnh:** `rollout_ep_len_mean.png` (Biểu đồ tương ứng với dữ liệu `csv.csv`).
> *   **Chú thích dưới hình:** *Hình 4.3: Biểu đồ tuổi thọ trung bình của Agent (rollout/ep_len_mean) ghi nhận từ Tensorboard, thể hiện quá trình sinh tồn kéo dài từ ~700 lên tới hơn 1,034 frames.*""":
    """![Hình 4.3: Biểu đồ tuổi thọ trung bình của Agent (rollout/ep_len_mean)](rollout_ep_len_mean.png)
*Hình 4.3: Biểu đồ tuổi thọ trung bình của Agent (rollout/ep_len_mean) ghi nhận từ Tensorboard, thể hiện quá trình sinh tồn kéo dài từ ~700 lên tới hơn 1,034 frames.*""",

    # 4. FPS
    """> [!IMPORTANT]
> **[CHÈN ẢNH BIỂU ĐỒ FPS TẠI ĐÂY - Dành cho Mục 4.2.2]**
> *   **Tên ảnh:** `time_fps.png` (Biểu đồ thứ 1/3 người dùng tải lên).
> *   **Chú thích dưới hình:** *Hình 4.4: Biểu đồ tốc độ khung hình (time/fps) ghi nhận từ Tensorboard, thể hiện hiệu năng xử lý dao động ổn định trong khoảng 1500 - 2100 FPS.*""":
    """![Hình 4.4: Biểu đồ tốc độ khung hình (time/fps)](time_fps.png)
*Hình 4.4: Biểu đồ tốc độ khung hình (time/fps) ghi nhận từ Tensorboard, thể hiện hiệu năng xử lý dao động ổn định trong khoảng 1500 - 2100 FPS.*""",

    # 5. KL Divergence
    """> [!IMPORTANT]
> **[CHÈN ẢNH BIỂU ĐỒ PHÂN KỲ KL TẠI ĐÂY]**
> *   **Tên ảnh:** `train_approx_kl.png`
> *   **Chú thích dưới hình:** *Hình 4.5: Biểu đồ phân kỳ KL (approx_kl) thể hiện độ sốc khi cập nhật phản xạ xác suất hành động qua các bước huấn luyện.*""":
    """![Hình 4.5: Biểu đồ phân kỳ KL (approx_kl)](train_approx_kl.png)
*Hình 4.5: Biểu đồ phân kỳ KL (approx_kl) thể hiện độ sốc khi cập nhật phản xạ xác suất hành động qua các bước huấn luyện.*""",

    # 6. Entropy Loss
    """> [!IMPORTANT]
> **[CHÈN ẢNH BIỂU ĐỒ ENTROPY LOSS TẠI ĐÂY]**
> *   **Tên ảnh:** `train_entropy_loss.png`
> *   **Chú thích dưới hình:** *Hình 4.6: Biểu đồ suy giảm Entropy (entropy_loss) thể hiện tốc độ chuyển đổi từ tò mò khám phá sang khai thác hành động tối ưu.*""":
    """![Hình 4.6: Biểu đồ suy giảm Entropy (entropy_loss)](train_entropy_loss.png)
*Hình 4.6: Biểu đồ suy giảm Entropy (entropy_loss) thể hiện tốc độ chuyển đổi từ tò mò khám phá sang khai thác hành động tối ưu.*""",

    # 7. Explained Variance
    """> [!IMPORTANT]
> **[CHÈN ẢNH BIỂU ĐỒ PHƯƠNG SAI GIẢI THÍCH TẠI ĐÂY]**
> *   **Tên ảnh:** `train_explained_variance.png`
> *   **Chú thích dưới hình:** *Hình 4.7: Biểu đồ phương sai giải thích (explained_variance) thể hiện độ thông minh và chính xác của mạng giá trị Critic.*""":
    """![Hình 4.7: Biểu đồ phương sai giải thích (explained_variance)](train_explained_variance.png)
*Hình 4.7: Biểu đồ phương sai giải thích (explained_variance) thể hiện độ thông minh và chính xác của mạng giá trị Critic.*""",

    # 8. Value Loss Comparison
    """> [!IMPORTANT]
> **[CHÈN ẢNH ĐỒ THỊ SO SÁNH VALUE LOSS TẠI ĐÂY]**
> *   **Tên ảnh:** `value_loss_comparison.png`
> *   **Chú thích dưới hình:** *Hình 4.8: Đồ thị so sánh trị số value_loss giữa cấu hình huấn luyện trực tiếp không curriculum (nổ loss lên 12.1) và cấu hình tịnh tiến curriculum (ổn định ở mức 1.56).*""":
    """![Hình 4.8: Đồ thị so sánh trị số value_loss giữa cấu hình huấn luyện trực tiếp không curriculum và cấu hình tịnh tiến curriculum](value_loss_comparison.png)
*Hình 4.8: Đồ thị so sánh trị số value_loss giữa cấu hình huấn luyện trực tiếp không curriculum (nổ loss lên 12.1) và cấu hình tịnh tiến curriculum (ổn định ở mức 1.56).*""",

    # 9. Gameplay 1 & 2
    """> [!IMPORTANT]
> **[CHÈN ẢNH CHỤP GAMEPLAY STAGE 1 & 2 TẠI ĐÂY]**
> *   **Tên ảnh:** `gameplay_stage_1_2.png`
> *   **Chú thích dưới hình:** *Hình 4.9: Giao diện game thực tế tại Stage 1 (Boss đứng im để Agent tập bắn) và Stage 2 (Agent bắt đầu di chuyển né tránh các làn đạn thẳng cơ bản).*""":
    """![Hình 4.9: Giao diện game thực tế tại Stage 1 và Stage 2](gameplay_stage_1_2.png)
*Hình 4.9: Giao diện game thực tế tại Stage 1 (Boss đứng im để Agent tập bắn) và Stage 2 (Agent bắt đầu di chuyển né tránh các làn đạn thẳng cơ bản).*""",

    # 10. Gameplay 3 & 4
    """> [!IMPORTANT]
> **[CHÈN ẢNH CHỤP GAMEPLAY STAGE 3 & 4 TẠI ĐÂY]**
> *   **Tên ảnh:** `gameplay_stage_3_4.png`
> *   **Chú thích dưới hình:** *Hình 4.10: Giao diện game thực tế tại Stage 3 (Boss di chuyển qua lại) và Stage 4 (Boss xả đạn chùm và đạn đuổi khép góc cực kỳ nguy hiểm).*""":
    """![Hình 4.10: Giao diện game thực tế tại Stage 3 và Stage 4](gameplay_stage_3_4.png)
*Hình 4.10: Giao diện game thực tế tại Stage 3 (Boss di chuyển qua lại) và Stage 4 (Boss xả đạn chùm và đạn đuổi khép góc cực kỳ nguy hiểm).*"""
}

# Perform replacements
for old_block, new_block in replacements.items():
    if old_block in content:
        content = content.replace(old_block, new_block)
    else:
        print(f"Warning: A placeholder block was not found in the original file.")

with open(dest_file, 'w', encoding='utf-8') as f:
    f.write(content)

print(f"Successfully generated consolidated report at: {dest_file}")
