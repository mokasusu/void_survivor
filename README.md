# Void Survivor

Game bắn súng top-down được phát triển bằng Python và Pygame.

## Cách chạy

```bash
python -m venv venv
venv\Scripts\activate
pip install pygame
python main.py
```

Hoặc chạy file `run.bat` trên Windows.

## Điều khiển

| Phím | Chức năng |
|------|-----------|
| ← / → / ↑ / ↓ | Di chuyển |
| SPACE | Bắn |
| ESC | Quay lại menu |
| R | Chơi lại |

## Chế độ chơi

- **Survival**: Né đạn và sống càng lâu càng tốt
- **Boss**: Đánh bại boss 100 HP

## Tính năng mới

- Di chuyển trái/phải bằng mũi tên ← →
- Khi người chơi trúng đạn của boss hoặc va chạm với boss, người chơi sẽ có thời gian bất tử 3 giây (nhấp nháy, không nhậnDamage)

## Cấu trúc thư mục

```
void_survivor/
├── main.py               # Điểm khởi chạy
├── run.bat               # Chạy nhanh trên Windows
├── requirements.txt      # Phụ thuộc
├── .gitignore
├── assets/               # Hình ảnh, sprite
├── config/
│   └── settings.py       # Cài đặt game
├── controllers/
│   ├── action.py         # Enum hành động
│   └── human_controller.py  # Input bàn phím
├── core/
│   ├── game.py           # Logic chính
│   ├── ui.py             # Giao diện (HP, timer)
│   ├── assets.py         # Load hình ảnh
│   └── collision.py      # Phát hiện va chạm
├── entities/
│   ├── player.py         # Nhân vật người chơi
│   ├── boss.py           # Boss
│   ├── bullet.py         # Đạn kẻ địch
│   └── player_bullet.py  # Đạn người chơi
├── managers/
│   ├── bullet_manager.py
│   └── player_bullet_manager.py
├── scenes/
│   ├── menu_scene.py     # Menu chính
│   ├── mode_scene.py     # Chọn chế độ
│   ├── guide_scene.py    # Hướng dẫn
│   └── game_scene.py     # Gameplay + Game Over
└── ui/
    └── button.py         # Thành phần Button
```

## Cấu hình

Xem `config/settings.py` để tùy chỉnh: kích thước màn hình (800x600), FPS (60), màu sắc, v.v.

## Yêu cầu

- Python 3.10+
- Pygame (`pip install pygame`)