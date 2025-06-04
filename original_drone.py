import pygame
import sys

# Initialize Pypygame
pygame.init()
pygame.joystick.init()

# Screen dimensions
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Xbox Controller Input Viewer")

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

# Box properties
box_width = 50
box_height = 50
# Initial position of the box (center of the screen)
pos_x = SCREEN_WIDTH / 2 - box_width / 2
pos_y = SCREEN_HEIGHT / 2 - box_height / 2
box_speed = 300  # Pixels per second

# Joystick variables
left_x = 0.0
left_y = 0.0
right_y = 0.0

# Font for displaying text
font = pygame.font.Font(None, 36)

# Clock for managing frame rate
clock = pygame.time.Clock()
FPS = 60  # Frames per second

print(pygame.joystick.get_count())
# print(f"Detected joystick: {joystick.get_name()}")

# Joystick setup
joystick_count = pygame.joystick.get_count()
if joystick_count == 0:
    print("Error: No joysticks/controllers detected.")
    pygame.quit()
    sys.exit()
else:
    # Use the first joystick found
    joystick = pygame.joystick.Joystick(0)
    joystick.init()
    print(f"Initialized Joystick: {joystick.get_name()}")
    # Check for number of axes (Xbox controllers typically have at least 5)
    if joystick.get_numaxes() < 5:
        print(f"Warning: Joystick {joystick.get_name()} has fewer than 5 axes. Axis mapping might be incorrect.")


# --- Main pygame Loop ---
running = True
while running:
    # Calculate delta time (dt) in seconds
    # This ensures movement speed is consistent regardless of frame rate
    dt = clock.tick(FPS) / 1000.0

    # --- Event Handling ---
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        # Joystick events are usually handled by polling axes, not specific events for axis motion
        # However, button presses/releases can be caught here if needed
        # For example:
        # if event.type == pypygame.JOYBUTTONDOWN:
        #     print(f"Joystick button {event.button} pressed.")
        # if event.type == pypygame.JOYAXISMOTION:
        #     # This event also fires, but polling is often simpler for continuous input
        #     pass


    # --- Joystick Input ---
    if joystick:
        # Axis 0: Left Stick X (-1.0 to 1.0, left to right)
        # Axis 1: Left Stick Y (-1.0 to 1.0, up to down)
        # Axis 4: Right Stick Y (-1.0 to 1.0, up to down) - Note: This can vary (sometimes Axis 3)
        # You might need to adjust axis indices based on your controller and OS.
        # Add a small deadzone to prevent drift from slight imperfections in the joystick
        dead_zone = 0.1

        current_left_x = joystick.get_axis(0)
        if abs(current_left_x) > dead_zone:
            left_x = current_left_x
        else:
            left_x = 0.0

        current_left_y = joystick.get_axis(1)
        if abs(current_left_y) > dead_zone:
            left_y = current_left_y
        else:
            left_y = 0.0
        
        # Attempt to get Right Stick Y. Common axes are 4 or 3.
        try:
            current_right_y = joystick.get_axis(4) # Standard for many XInput controllers
        except pygame.error:
            try:
                current_right_y = joystick.get_axis(3) # Alternative
            except pygame.error:
                print("Could not read right_y axis (tried 3 and 4). Ensure controller has enough axes.")
                current_right_y = 0.0 # Default if not found

        if abs(current_right_y) > dead_zone:
            right_y = current_right_y
        else:
            right_y = 0.0

    # --- pygame Logic / Update State ---
    # Update box position based on left joystick input and delta time
    pos_x += left_x * box_speed * dt
    pos_y += left_y * box_speed * dt

    # Keep box within screen bounds
    if pos_x < 0:
        pos_x = 0
    elif pos_x > SCREEN_WIDTH - box_width:
        pos_x = SCREEN_WIDTH - box_width
    
    if pos_y < 0:
        pos_y = 0
    elif pos_y > SCREEN_HEIGHT - box_height:
        pos_y = SCREEN_HEIGHT - box_height

    # --- Drawing ---
    screen.fill(BLACK)  # Clear the screen

    # Draw the box
    pygame.draw.rect(screen, RED, (pos_x, pos_y, box_width, box_height))

    # Display joystick values
    text_left_x = font.render(f"Left X: {left_x:.2f}", True, WHITE)
    text_left_y = font.render(f"Left Y: {left_y:.2f}", True, WHITE)
    text_right_y = font.render(f"Right Y: {right_y:.2f}", True, WHITE)
    
    screen.blit(text_left_x, (10, 10))
    screen.blit(text_left_y, (10, 50))
    screen.blit(text_right_y, (10, 90))

    # Update the full display
    pygame.display.flip()

# --- Cleanup ---
pygame.joystick.quit()
pygame.quit()
sys.exit()
