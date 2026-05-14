import pygame
from config.settings import WIDTH

class Laser:
    def __init__(self, start_x, y, height=40):
        self.start_x = start_x
        self.y = y
        self.height = height
        
        self.warning_frames = 60
        self.active_frames = 30
        self.timer = 0
        
        self.state = "warning"

    def update(self, start_x, y):
        self.start_x = start_x
        self.y = y
        self.timer += 1
        
        if self.state == "warning" and self.timer >= self.warning_frames:
            self.state = "active"
            self.timer = 0
        elif self.state == "active" and self.timer >= self.active_frames:
            self.state = "done"

    def get_damage_rect(self):
        if self.state == "active":
            return pygame.Rect(self.start_x, self.y - self.height // 2, WIDTH - self.start_x, self.height)
        return None

    def draw(self, screen):
        if self.state == "warning":
            pygame.draw.line(screen, (255, 100, 100), (self.start_x, self.y), (WIDTH, self.y), 2)
        elif self.state == "active":
            rect = self.get_damage_rect()
            if rect:
                pygame.draw.rect(screen, (255, 255, 255), rect)
                aura_rect = pygame.Rect(self.start_x, self.y - self.height // 2 - 5, WIDTH - self.start_x, self.height + 10)
                surface = pygame.Surface((aura_rect.width, aura_rect.height), pygame.SRCALPHA)
                surface.fill((255, 50, 50, 120))
                screen.blit(surface, (aura_rect.x, aura_rect.y))
