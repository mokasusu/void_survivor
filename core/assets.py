import os
import pygame


ASSETS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "assets"
)


def load_image(filename, size=None):
    path = os.path.join(ASSETS_DIR, filename)

    try:
        image = pygame.image.load(path).convert_alpha()
    except Exception:
        return None

    if size is not None:
        image = pygame.transform.smoothscale(image, size)

    return image
