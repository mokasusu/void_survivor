import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

output_dir = "/Users/vducc3110/Desktop/void_survivor"
os.makedirs(output_dir, exist_ok=True)

def draw_model_zip_structure():
    fig, ax = plt.subplots(figsize=(8, 4), dpi=300)
    ax.axis('off')
    
    # Draw ZIP container
    container = patches.FancyBboxPatch((0.05, 0.1), 0.9, 0.8, boxstyle="round,pad=0.03", 
                                      facecolor='#F3F4F6', edgecolor='#9CA3AF', linewidth=2)
    ax.add_patch(container)
    ax.text(0.5, 0.85, "Checkpoint Model (.zip)", fontsize=14, fontweight='bold', ha='center', color='#1F2937')
    
    # Draw sub-files
    files = [
        ("data (JSON)", "Lưu trữ cấu hình thuật toán,\nsiêu tham số (learning_rate,\nent_coef, gamma) và định nghĩa\nkhông gian State/Action.", '#DBEAFE', '#2563EB'),
        ("parameter_list", "Danh sách định nghĩa kiểu\ndữ liệu và kích thước (shape)\ncủa các lớp tensor trọng số.", '#FEE2E2', '#DC2626'),
        ("policy.pt", "Trọng số thực tế (weights & biases)\ncủa mạng Actor và mạng Critic\ndưới dạng lưu trữ PyTorch.", '#D1FAE5', '#059669'),
        ("policy.optimizer.pt", "Trạng thái thuật toán tối ưu hóa\nAdam (momen động lượng, learning\nrate thích nghi) để chạy tiếp train.", '#FEF3C7', '#D97706')
    ]
    
    for idx, (name, desc, bg, border) in enumerate(files):
        x = 0.08 + idx * 0.21
        y = 0.18
        w = 0.20
        h = 0.60
        
        box = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01", 
                                     facecolor=bg, edgecolor=border, linewidth=1.5)
        ax.add_patch(box)
        ax.text(x + w/2, y + h - 0.08, name, fontsize=10, fontweight='bold', ha='center', color='#111827')
        ax.text(x + w/2, y + h - 0.32, desc, fontsize=7.5, ha='center', va='center', color='#374151', wrap=True)

    plt.tight_layout()
    path = os.path.join(output_dir, "model_zip_structure.png")
    plt.savefig(path, bbox_inches='tight', facecolor='#FFFFFF')
    plt.close()
    print(f"Generated: {path}")

def draw_inference_flowchart():
    fig, ax = plt.subplots(figsize=(10, 3.5), dpi=300)
    ax.axis('off')
    
    # Define steps
    steps = [
        "1. Trạng thái Game\n(Vị trí phi thuyền,\nBoss, 10 viên đạn)",
        "2. PPOStateEncoder\n(Chuẩn hóa thành\nvector 45 chiều)",
        "3. Action Masking\n(Khóa các hành động\nbắn khi nạp đạn)",
        "4. Model predict\n( deterministic=True\nxuất Action tối ưu)",
        "5. Game Loop\n(Thực thi hành động,\nvẽ màn hình 60 FPS)"
    ]
    
    # Draw boxes and arrows
    for idx, text in enumerate(steps):
        x = 0.05 + idx * 0.19
        y = 0.25
        w = 0.15
        h = 0.50
        
        box = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02", 
                                     facecolor='#EFF6FF', edgecolor='#3B82F6', linewidth=2)
        ax.add_patch(box)
        ax.text(x + w/2, y + h/2, text, fontsize=9, fontweight='bold', ha='center', va='center', color='#1E3A8A')
        
        # Draw arrow to next block
        if idx < len(steps) - 1:
            ax.annotate('', xy=(x + w + 0.035, y + h/2), xytext=(x + w + 0.005, y + h/2),
                        arrowprops=dict(arrowstyle="->", color='#3B82F6', lw=2.5))
            
    plt.tight_layout()
    path = os.path.join(output_dir, "inference_flowchart.png")
    plt.savefig(path, bbox_inches='tight', facecolor='#FFFFFF')
    plt.close()
    print(f"Generated: {path}")

def draw_game_interface_layout():
    fig, ax = plt.subplots(figsize=(7, 5.5), dpi=300)
    ax.set_xlim(0, 800)
    ax.set_ylim(0, 600)
    ax.invert_yaxis() # Pygame coordinate system (0,0 at top-left)
    
    # Background
    ax.set_facecolor('#1E1E1E')
    fig.patch.set_facecolor('#1E1E1E')
    
    # Grid lines to resemble space grids
    ax.grid(True, color='#2D2D2D', linestyle='--', linewidth=0.5)
    
    # HUD: Boss HP (Top)
    boss_hp_bg = patches.Rectangle((100, 25), 600, 15, facecolor='#450A0A', edgecolor='#EF4444', linewidth=1)
    boss_hp = patches.Rectangle((100, 25), 450, 15, facecolor='#EF4444')
    ax.add_patch(boss_hp_bg)
    ax.add_patch(boss_hp)
    ax.text(400, 20, "BOSS HP: 75%", color='#EF4444', fontsize=9, fontweight='bold', ha='center')
    
    # HUD: Player HP (Top Left)
    player_hp_bg = patches.Rectangle((20, 55), 150, 12, facecolor='#064E3B', edgecolor='#10B981', linewidth=1)
    player_hp = patches.Rectangle((20, 55), 120, 12, facecolor='#10B981')
    ax.add_patch(player_hp_bg)
    ax.add_patch(player_hp)
    ax.text(20, 50, "PLAYER HP", color='#10B981', fontsize=8, fontweight='bold')
    
    # Boss Sprite Area
    boss = patches.Rectangle((320, 80), 160, 60, facecolor='#374151', edgecolor='#9CA3AF', hatch='//')
    ax.add_patch(boss)
    ax.text(400, 115, "BOSS\n(Stage 4)", color='#FFFFFF', fontsize=10, fontweight='bold', ha='center', va='center')
    
    # Player Center (cx=400, cy=420)
    pcx, pcy = 400, 420
    
    # Player Sprite Area
    player = patches.RegularPolygon((pcx, pcy), 3, radius=18, orientation=0, facecolor='#059669', edgecolor='#34D399')
    ax.add_patch(player)
    
    # Hitbox (15px)
    hitbox = patches.Circle((pcx, pcy), 15, fill=False, edgecolor='#EF4444', linestyle='-', linewidth=1.5, label='Hitbox (15px)')
    ax.add_patch(hitbox)
    ax.plot(pcx, pcy, 'ro', markersize=3)
    ax.text(pcx + 18, pcy + 4, "Hitbox (15px)", color='#EF4444', fontsize=8, fontweight='bold')
    
    # Grazing Zone (50px)
    grazing = patches.Circle((pcx, pcy), 50, fill=False, edgecolor='#60A5FA', linestyle='--', linewidth=1.5, label='Grazing Zone (50px)')
    ax.add_patch(grazing)
    ax.text(pcx + 55, pcy + 30, "Vùng Grazing\n(Bán kính 50px)", color='#60A5FA', fontsize=8, fontweight='bold')
    
    # Enemy Bullets
    bullets = [(380, 300), (420, 320), (450, 400), (360, 440)] # bullet positions
    for bx, by in bullets:
        bullet_outer = patches.Circle((bx, by), 8, fill=True, color='#EF4444', alpha=0.9)
        bullet_inner = patches.Circle((bx, by), 4, fill=True, color='#FFFFFF')
        ax.add_patch(bullet_outer)
        ax.add_patch(bullet_inner)
        
    # Draw grazing interaction line
    # Distance from player (400, 420) to bullet (450, 400) is sqrt(50^2 + 20^2) = 53.8px (approx)
    # Let's place one bullet exactly inside grazing zone, say (420, 440) -> distance sqrt(20^2 + 20^2) = 28.2px
    g_bx, g_by = 430, 435
    bullet_outer = patches.Circle((g_bx, g_by), 8, fill=True, color='#EF4444')
    bullet_inner = patches.Circle((g_bx, g_by), 4, fill=True, color='#FFFFFF')
    ax.add_patch(bullet_outer)
    ax.add_patch(bullet_inner)
    ax.plot([pcx, g_bx], [pcy, g_by], color='#60A5FA', linestyle=':', linewidth=1.5)
    ax.text((pcx+g_bx)/2 - 5, (pcy+g_by)/2 - 8, "d=32px", color='#60A5FA', fontsize=7, fontweight='bold')
    ax.text(g_bx + 10, g_by - 5, "Ăn điểm Grazing\n(+0.05 reward)", color='#34D399', fontsize=7.5, fontweight='bold')
    
    # Info Panel (Bottom 90px)
    info_panel = patches.Rectangle((0, 510), 800, 90, facecolor='#111827', edgecolor='#374151', linewidth=1.5)
    ax.add_patch(info_panel)
    ax.text(50, 545, "STAGE: 4", color='#00D2FF', fontsize=10, fontweight='bold')
    ax.text(50, 570, "TIME: 142.5s", color='#00D2FF', fontsize=10, fontweight='bold')
    ax.text(320, 545, "REWARD: +248.55", color='#00D2FF', fontsize=10, fontweight='bold')
    ax.text(320, 570, "GRAZE COUNT: 149", color='#00D2FF', fontsize=10, fontweight='bold')
    ax.text(600, 555, "COOLDOWN: 0/6", color='#00D2FF', fontsize=10, fontweight='bold')
    ax.text(400, 500, "BẢNG THÔNG SỐ HUD (INFO_PANEL_HEIGHT = 90px)", color='#9CA3AF', fontsize=8, ha='center')

    # Border for the window
    window_border = patches.Rectangle((0, 0), 800, 600, fill=False, edgecolor='#4B5563', linewidth=3)
    ax.add_patch(window_border)
    
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    for spine in ax.spines.values():
        spine.set_visible(False)
        
    plt.tight_layout()
    path = os.path.join(output_dir, "game_interface_layout.png")
    plt.savefig(path, bbox_inches='tight', facecolor='#1E1E1E')
    plt.close()
    print(f"Generated: {path}")

if __name__ == "__main__":
    draw_model_zip_structure()
    draw_inference_flowchart()
    draw_game_interface_layout()
