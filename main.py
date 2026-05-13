import pygame

from config.settings import WIDTH
from config.settings import HEIGHT
from config.settings import FPS

from scenes.menu_scene import MenuScene


class MainGame:

    def __init__(self):

        pygame.init()

        self.screen = pygame.display.set_mode(
            (WIDTH, HEIGHT)
        )

        pygame.display.set_caption(
            "VOID SURVIVOR"
        )

        self.clock = pygame.time.Clock()

        self.running = True
        self.selected_mode = "survival"

        self.current_scene = MenuScene(
            self
        )

    def run(self):

        while self.running:

            self.clock.tick(FPS)

            for event in pygame.event.get():

                if event.type == pygame.QUIT:

                    self.running = False

                self.current_scene.handle_event(
                    event
                )

            self.current_scene.update()

            self.current_scene.draw(
                self.screen
            )

            pygame.display.flip()

        pygame.quit()


if __name__ == "__main__":

    game = MainGame()

    game.run()
