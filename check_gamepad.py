import pygame

pygame.init()
pygame.joystick.init()

count = pygame.joystick.get_count()
print("Detected joysticks:", count)

for i in range(count):
    js = pygame.joystick.Joystick(i)
    js.init()
    print(f"[{i}] name = '{js.get_name()}'")
