#!/usr/bin/env python3
"""Simple pygame showcase with moving blocks and local images."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path

try:
    import pygame
except ImportError as exc:  # pragma: no cover - runtime dependency check
    raise SystemExit(
        "pygame 未安装。请先执行: python3 -m pip install pygame"
    ) from exc

try:
    from PIL import Image
except ImportError:
    Image = None


WINDOW_WIDTH = 1100
WINDOW_HEIGHT = 680
FPS = 60
PLAYGROUND_WIDTH = 760
IMAGE_PANEL_X = 790


@dataclass
class MovingBlock:
    x: float
    y: float
    size: int
    vx: float
    vy: float
    color: tuple[int, int, int]
    spin: float
    angle: float = 0.0

    def update(self, dt: float) -> None:
        self.x += self.vx * dt
        self.y += self.vy * dt
        self.angle = (self.angle + self.spin * dt) % 360

        if self.x <= 20 or self.x + self.size >= PLAYGROUND_WIDTH - 20:
            self.vx *= -1
            self.x = max(20, min(self.x, PLAYGROUND_WIDTH - 20 - self.size))
        if self.y <= 80 or self.y + self.size >= WINDOW_HEIGHT - 30:
            self.vy *= -1
            self.y = max(80, min(self.y, WINDOW_HEIGHT - 30 - self.size))

    def draw(self, surface: pygame.Surface) -> None:
        square = pygame.Surface((self.size, self.size), pygame.SRCALPHA)
        pygame.draw.rect(square, (*self.color, 220), (0, 0, self.size, self.size), border_radius=12)
        pygame.draw.rect(square, (255, 255, 255, 100), (5, 5, self.size - 10, self.size - 10), width=2, border_radius=10)
        rotated = pygame.transform.rotate(square, self.angle)
        rect = rotated.get_rect(center=(self.x + self.size / 2, self.y + self.size / 2))
        surface.blit(rotated, rect)


@dataclass
class FloatingImage:
    surface: pygame.Surface
    x: float
    base_y: float
    speed: float
    phase: float

    def update(self, elapsed: float) -> None:
        self.phase += self.speed * elapsed

    def draw(self, surface: pygame.Surface) -> None:
        y = self.base_y + math.sin(self.phase) * 18
        shadow = pygame.Surface((self.surface.get_width() + 20, self.surface.get_height() + 20), pygame.SRCALPHA)
        pygame.draw.rect(
            shadow,
            (0, 0, 0, 70),
            shadow.get_rect(),
            border_radius=20,
        )
        surface.blit(shadow, (self.x - 10, y - 10))
        surface.blit(self.surface, (self.x, y))
        marker_rect = pygame.Rect(self.x, y + self.surface.get_height() + 14, 120, 8)
        pygame.draw.rect(surface, (238, 245, 255), marker_rect, border_radius=4)
        pygame.draw.rect(surface, (125, 211, 252), (self.x, y + self.surface.get_height() + 14, 54, 8), border_radius=4)


def load_showcase_images() -> list[FloatingImage]:
    image_dir = Path(__file__).resolve().parent.parent / "data" / "images" / "tableturf_full"
    image_paths = sorted(p for p in image_dir.glob("*.png"))[:3]
    images: list[FloatingImage] = []
    for index, path in enumerate(image_paths):
        raw = load_image_surface(path)
        image = pygame.transform.smoothscale(raw, (220, 120))
        images.append(
            FloatingImage(
                surface=image,
                x=IMAGE_PANEL_X,
                base_y=100 + index * 180,
                speed=1.1 + index * 0.35,
                phase=index * 0.9,
            )
        )
    return images


def load_image_surface(path: Path) -> pygame.Surface:
    try:
        return pygame.image.load(path.as_posix()).convert_alpha()
    except pygame.error:
        if Image is None:
            raise SystemExit(
                "当前 pygame 不支持 PNG 解码，且 Pillow 未安装。请执行: .venv/bin/python -m pip install pillow"
            )
        pil_image = Image.open(path).convert("RGBA")
        return pygame.image.fromstring(
            pil_image.tobytes(),
            pil_image.size,
            pil_image.mode,
        ).convert_alpha()


def create_blocks() -> list[MovingBlock]:
    palette = [
        (255, 99, 72),
        (46, 204, 113),
        (52, 152, 219),
        (241, 196, 15),
        (155, 89, 182),
        (26, 188, 156),
        (230, 126, 34),
    ]
    blocks: list[MovingBlock] = []
    for _ in range(9):
        size = random.randint(36, 92)
        blocks.append(
            MovingBlock(
                x=random.randint(30, PLAYGROUND_WIDTH - size - 30),
                y=random.randint(100, WINDOW_HEIGHT - size - 30),
                size=size,
                vx=random.choice([-1, 1]) * random.uniform(120, 240),
                vy=random.choice([-1, 1]) * random.uniform(90, 200),
                color=random.choice(palette),
                spin=random.uniform(-90, 90),
            )
        )
    return blocks


def draw_background(surface: pygame.Surface, elapsed: float) -> None:
    surface.fill((11, 18, 32))
    for y in range(WINDOW_HEIGHT):
        blend = y / WINDOW_HEIGHT
        color = (
            int(12 + 40 * blend),
            int(20 + 25 * blend),
            int(40 + 50 * blend),
        )
        pygame.draw.line(surface, color, (0, y), (WINDOW_WIDTH, y))

    for index in range(14):
        radius = 20 + index * 12
        x = 150 + index * 40 + math.sin(elapsed * 0.8 + index) * 28
        y = 40 + math.cos(elapsed * 0.7 + index * 0.4) * 16
        pygame.draw.circle(surface, (255, 255, 255, 18), (int(x), int(y)), radius, width=1)

    pygame.draw.rect(surface, (18, 27, 46), (0, 0, PLAYGROUND_WIDTH, WINDOW_HEIGHT), border_radius=0)
    pygame.draw.line(surface, (90, 121, 170), (PLAYGROUND_WIDTH, 40), (PLAYGROUND_WIDTH, WINDOW_HEIGHT - 40), 2)


def draw_hud(surface: pygame.Surface, blocks: list[MovingBlock], elapsed: float) -> None:
    header = pygame.Rect(28, 24, 360, 74)
    pygame.draw.rect(surface, (255, 255, 255, 18), header, border_radius=18)
    pygame.draw.rect(surface, (255, 255, 255, 52), header, width=1, border_radius=18)
    pygame.draw.rect(surface, (125, 211, 252), (44, 42, 160, 12), border_radius=6)
    pygame.draw.rect(surface, (244, 114, 182), (44, 64, 250, 8), border_radius=4)
    pygame.draw.rect(surface, (74, 222, 128), (44, 80, max(36, len(blocks) * 28), 6), border_radius=3)
    pulse_x = 320 + math.sin(elapsed * 2.6) * 16
    pygame.draw.circle(surface, (250, 204, 21), (int(pulse_x), 61), 10)

    panel = pygame.Surface((280, WINDOW_HEIGHT - 60), pygame.SRCALPHA)
    pygame.draw.rect(panel, (255, 255, 255, 22), panel.get_rect(), border_radius=24)
    pygame.draw.rect(panel, (255, 255, 255, 45), panel.get_rect(), width=1, border_radius=24)
    surface.blit(panel, (IMAGE_PANEL_X - 20, 30))
    for index in range(3):
        y = 54 + index * 180
        pygame.draw.rect(surface, (255, 255, 255, 40), (IMAGE_PANEL_X + 150, y, 80, 5), border_radius=2)
        pygame.draw.rect(surface, (125, 211, 252), (IMAGE_PANEL_X + 150, y + 12, 56, 5), border_radius=2)


def main() -> int:
    pygame.init()
    pygame.display.set_caption("Pygame Showcase")
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    clock = pygame.time.Clock()

    blocks = create_blocks()
    images = load_showcase_images()
    elapsed = 0.0
    running = True

    while running:
        dt = clock.tick(FPS) / 1000.0
        elapsed += dt

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False

        for block in blocks:
            block.update(dt)
        for image in images:
            image.update(dt)

        draw_background(screen, elapsed)
        draw_hud(screen, blocks, elapsed)

        for block in blocks:
            block.draw(screen)
        for image in images:
            image.draw(screen)

        pygame.display.flip()

    pygame.quit()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
