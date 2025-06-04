import numpy as np
import time
import os
import pygame
import sys
import tensorflow as tf # Added for loading the model

pygame.init()
pygame.joystick.init()

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

# Clock for managing frame rate
font = pygame.font.Font(None, 15) # Adjusted font size for more text
clock = pygame.time.Clock()
FPS = 60  # Frames per second

# --- ML Model Integration ---
# Load the trained model
try:
    model = tf.keras.models.load_model('predictor_model.keras')
    print("Successfully loaded predictor_model.keras")
except Exception as e:
    print(f"Error loading model predictor_model.keras: {e}")
    pygame.quit()
    sys.exit()

# Custom StandardScaler class (as used in the notebook)
class StandardScaler:
    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, X):
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)
        self.std_[self.std_ == 0] = 1.0  # Avoid division by zero
        return self

    def transform(self, X):
        if self.mean_ is None or self.std_ is None:
            raise ValueError("Scaler has not been fitted yet.")
        return (X - self.mean_) / self.std_

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X):
        if self.mean_ is None or self.std_ is None:
            raise ValueError("Scaler has not been fitted yet.")
        return X * self.std_ + self.mean_

scaler_X = StandardScaler()
scaler_y = StandardScaler()

# !!! IMPORTANT: Replace these placeholder values with actual mean and std
# !!! from your training data used for 'predictor_model.keras'.
# !!! The order of features in mean_ and std_ must match the training:
# !!! ['OSD.pitch', 'OSD.roll', 'OSD.yaw', 'OSD.xSpeed [MPH]', 'OSD.ySpeed [MPH]', 'OSD.zSpeed [MPH]',
# !!!  'RC.aileron', 'RC.elevator', 'RC.throttle', 'RC.rudder']
scaler_X.mean_ = np.array([ 6.10798122e-01, -8.85269953e-01,  9.47972418e+00, -2.04357071e+00,
        4.19288410e+00, -7.31271964e-01,  9.24335974e+02,  1.06818134e+03,
        1.06898650e+03,  1.03743192e+03])  # Placeholder
scaler_X.std_  = np.array([ 19.13924807,   6.46803989, 131.59558227,  14.69475922,
        17.31924777,   2.85329585, 254.6548688 , 593.99587519,
       185.37528186, 134.97482834]) # Placeholder

# !!! Order for targets:
# !!! ['OSD.pitch', 'OSD.roll', 'OSD.yaw', 'OSD.xSpeed [MPH]', 'OSD.ySpeed [MPH]', 'OSD.zSpeed [MPH]']
scaler_y.mean_ = np.array([ 0.60642606, -0.88565141,  9.40936033, -2.04048573,  4.18638595,
       -0.73127196]) # Placeholder
scaler_y.std_  = np.array([ 19.14179691,   6.46786087, 131.5710077 ,  14.69629144,
        17.32497328,   2.85329585]) # Placeholder


# Initial state for the model-predicted drone
model_pitch = 0.0  # degrees
model_roll = 0.0   # degrees
model_yaw = 0.0    # degrees (radians for np.cos/sin, but pygame rotate needs degrees)
model_xSpeed = 0.0 # MPH
model_ySpeed = 0.0 # MPH
model_zSpeed = 0.0 # MPH

# Position for model-predicted drone
model_pos_x = SCREEN_WIDTH * 0.75 - box_width / 2
model_pos_y = SCREEN_HEIGHT / 2 - box_height / 2
model_altitude = 0.0 # Represents Z position, updated by model_zSpeed

# Conversion factor for speeds (example, tune this)
# 1 MPH = ~0.447 m/s. If 1m = 20 pixels, then 1 MPH = 0.447 * 20 = ~8.94 pixels/s
# Let's use a simpler factor for now, e.g., 1 MPH = 10 pixels/s for easier visualization
MPH_TO_PIXELS_PER_SEC = 10.0


start_time = time.time()




# dt = 0.01
rot_speed_factor = 1
yaw_speed_factor = 1
pos_x = 0
pos_y = 0
pos_z = 0
yaw = 0


running = True
# --- Reset Logic for Model Drone ---
RESET_INTERVAL_MS = 5000  # Reset every 5 seconds
last_reset_ticks = -RESET_INTERVAL_MS # Ensure first reset happens immediately
# Store current speeds of the original drone for reset reference
current_orig_vx_screen_pps = 0.0
current_orig_vy_screen_pps = 0.0
current_orig_vz_world_pps = 0.0
current_orig_yaw_rad = 0.0

while running:
    dt = clock.tick(FPS) / 1000.0


    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    dead_zone = 0.1

    # --- Original Drone Control (Manual Physics) ---
    current_left_x_orig = joystick.get_axis(0) # Typically Rudder/Yaw for Mode 2
    if abs(current_left_x_orig) > dead_zone:
        # This was mapped to roll in original script, let's keep its effect for the first box
        roll_input_orig = current_left_x_orig * 45. # degrees/sec
        left_x = roll_input_orig * rot_speed_factor * dt # effective change in roll for movement calc
    else:
        left_x = 0.0
    
    current_left_y_orig = joystick.get_axis(1) # Typically Throttle for Mode 2
    if abs(current_left_y_orig) > dead_zone:
        # This was mapped to pitch in original script
        pitch_input_orig = current_left_y_orig * 45. # degrees/sec
        left_y = pitch_input_orig * rot_speed_factor * dt # effective change in pitch for movement calc
    else:
        left_y = 0.0

    current_right_x_orig = joystick.get_axis(2) # Typically Aileron/Roll for Mode 2
    if abs(current_right_x_orig) > dead_zone:
        right_x = current_right_x_orig # Used for yaw update in original
    else:
        right_x = 0.0

    current_right_y_orig = joystick.get_axis(3) # Typically Elevator/Pitch for Mode 2
    if abs(current_right_y_orig) > dead_zone:
        right_y = current_right_y_orig # Used for z pos update in original
    else:
        right_y = 0.0
    
    # Original drone physics update
    cos_yaw_orig = np.cos(yaw)
    sin_yaw_orig = np.sin(yaw)

    # Original script's left_x and left_y were more like roll and pitch inputs affecting dx/dy
    # Let's assume left_x is sideways input (roll-like) and left_y is forward/backward (pitch-like)
    # For the original box, let's use right stick for X/Y movement and left stick X yaws, left stick Y moves Z
    # dx_orig = current_right_x_orig * cos_yaw_orig - (-current_right_y_orig) * sin_yaw_orig # Roll and Pitch from Right Stick
    # dy_orig = current_right_x_orig * sin_yaw_orig + (-current_right_y_orig) * cos_yaw_orig
    
    # Simpler mapping for original box: right stick X/Y moves it in screen space, left stick X yaws, left stick Y moves Z
    dx_orig = current_right_x_orig # X-movement from right stick X
    dy_orig = -current_right_y_orig # Y-movement from right stick Y (pygame Y is inverted)


    pos_x += dx_orig * box_speed * dt
    pos_y += dy_orig * box_speed * dt
    yaw += current_left_x_orig * yaw_speed_factor * dt # Yaw from left stick X
    pos_z += (-current_left_y_orig) * box_speed * dt # Z from left stick Y


    # --- Update for reset: original drone's X/Y/Z screen/world speeds and yaw
    if abs(current_right_x_orig) > dead_zone:
        current_orig_vx_screen_pps = current_right_x_orig * box_speed
    else:
        current_orig_vx_screen_pps = 0.0
    if abs(current_right_y_orig) > dead_zone:
        current_orig_vy_screen_pps = -current_right_y_orig * box_speed
    else:
        current_orig_vy_screen_pps = 0.0
    if abs(current_left_y_orig) > dead_zone:
        current_orig_vz_world_pps = -current_left_y_orig * box_speed
    else:
        current_orig_vz_world_pps = 0.0
    current_orig_yaw_rad = yaw

    # --- Model-Predicted Drone Control ---
    # Reset model drone every RESET_INTERVAL_MS
    current_ticks = pygame.time.get_ticks()
    if current_ticks - last_reset_ticks > RESET_INTERVAL_MS:
        # Set model drone's position to original drone's current position
        model_pos_x = pos_x
        model_pos_y = pos_y
        model_altitude = pos_z
        # Set model drone's orientation
        model_yaw = np.degrees(current_orig_yaw_rad)
        model_pitch = 0.0
        model_roll = 0.0
        # Calculate original drone's body-frame speeds from its screen/world speeds
        cos_orig_yaw = np.cos(current_orig_yaw_rad)
        sin_orig_yaw = np.sin(current_orig_yaw_rad)
        orig_body_xSpeed_pps = current_orig_vx_screen_pps * cos_orig_yaw + current_orig_vy_screen_pps * sin_orig_yaw
        orig_body_ySpeed_pps = -current_orig_vx_screen_pps * sin_orig_yaw + current_orig_vy_screen_pps * cos_orig_yaw
        model_xSpeed = orig_body_xSpeed_pps / MPH_TO_PIXELS_PER_SEC
        model_ySpeed = orig_body_ySpeed_pps / MPH_TO_PIXELS_PER_SEC
        model_zSpeed = current_orig_vz_world_pps / MPH_TO_PIXELS_PER_SEC
        last_reset_ticks = current_ticks

    # Get RC inputs for the model (standard Mode 2 mapping)
    # Values are typically -1 to 1 from joystick.get_axis()
    rc_aileron = joystick.get_axis(2)  # Right stick X (Roll)
    rc_elevator = -joystick.get_axis(3) # Right stick Y (Pitch, inverted: up is positive)
    rc_throttle = -joystick.get_axis(1) # Left stick Y (Throttle, inverted: up is positive)
    rc_rudder = joystick.get_axis(0)   # Left stick X (Yaw)

    # Apply dead zone
    rc_aileron = rc_aileron if abs(rc_aileron) > dead_zone else 0.0
    rc_elevator = rc_elevator if abs(rc_elevator) > dead_zone else 0.0
    rc_throttle = rc_throttle if abs(rc_throttle) > dead_zone else 0.0
    rc_rudder = rc_rudder if abs(rc_rudder) > dead_zone else 0.0

    # Construct input for the model
    # Order: OSD.pitch, OSD.roll, OSD.yaw, OSD.xSpeed, OSD.ySpeed, OSD.zSpeed, RC.aileron, RC.elevator, RC.throttle, RC.rudder
    model_input_features = np.array([
        model_pitch, model_roll, model_yaw,
        model_xSpeed, model_ySpeed, model_zSpeed,
        rc_aileron, rc_elevator, rc_throttle, rc_rudder
    ])

    # Scale input
    scaled_model_input = scaler_X.transform(model_input_features.reshape(1, -1))

    # Predict next state
    scaled_prediction = model.predict(scaled_model_input, verbose=0) # verbose=0 to suppress Keras print

    # Inverse scale prediction
    # Order: OSD.pitch, OSD.roll, OSD.yaw, OSD.xSpeed, OSD.ySpeed, OSD.zSpeed
    predicted_state = scaler_y.inverse_transform(scaled_prediction)

    model_pitch, model_roll, model_yaw, model_xSpeed, model_ySpeed, model_zSpeed = predicted_state[0]

    # Update model drone's 2D screen position based on predicted speeds
    # Assuming xSpeed is along drone's forward, ySpeed is to its right for simplicity in 2D projection
    # For a more accurate 2D projection from 3D speeds, you'd use pitch/roll/yaw to project to screen plane.
    # Here, let's assume xSpeed contributes to movement along current model_yaw, ySpeed perpendicular.
    
    # Convert speeds (MPH) to pixel displacement
    # Note: OSD.xSpeed is usually forward/backward, OSD.ySpeed is left/right, OSD.zSpeed is up/down in drone's body frame.
    # We need to translate these to world frame changes for screen coordinates.
    
    # Simplified: Treat model_xSpeed as forward, model_ySpeed as sideways relative to drone's yaw
    delta_forward = model_xSpeed * MPH_TO_PIXELS_PER_SEC * dt
    delta_sideways = model_ySpeed * MPH_TO_PIXELS_PER_SEC * dt # Positive ySpeed could be right

    model_pos_x += (delta_forward * np.cos(np.radians(model_yaw)) - delta_sideways * np.sin(np.radians(model_yaw)))
    model_pos_y += (delta_forward * np.sin(np.radians(model_yaw)) + delta_sideways * np.cos(np.radians(model_yaw)))
    
    # Update model altitude (not directly visualized as Z on the 2D box, but value is tracked)
    model_altitude += model_zSpeed * MPH_TO_PIXELS_PER_SEC * dt # MPH_TO_PIXELS_PER_SEC is a bit of a misnomer for altitude here, but scales it.


    # Keep original box within screen bounds
    if pos_x < 0:
        pos_x = 0
    elif pos_x > SCREEN_WIDTH - box_width:
        pos_x = SCREEN_WIDTH - box_width
    
    if pos_y < 0:
        pos_y = 0
    elif pos_y > SCREEN_HEIGHT - box_height:
        pos_y = SCREEN_HEIGHT - box_height

    # Keep model-predicted box within screen bounds
    if model_pos_x < 0: model_pos_x = 0
    elif model_pos_x > SCREEN_WIDTH - box_width: model_pos_x = SCREEN_WIDTH - box_width
    if model_pos_y < 0: model_pos_y = 0
    elif model_pos_y > SCREEN_HEIGHT - box_height: model_pos_y = SCREEN_HEIGHT - box_height


    # --- Drawing ---
    screen.fill(BLACK)  # Clear the screen

    # Draw the original box rotated based on yaw
    box_rect_orig = pygame.Rect(pos_x, pos_y, box_width, box_height)
    box_surf_orig = pygame.Surface((box_width, box_height), pygame.SRCALPHA)
    box_surf_orig.fill(RED) # Original box is RED
    rotated_surf_orig = pygame.transform.rotate(box_surf_orig, -np.degrees(yaw)) # yaw is already in radians from original code, convert for pygame
    rotated_rect_orig = rotated_surf_orig.get_rect(center=box_rect_orig.center)
    screen.blit(rotated_surf_orig, rotated_rect_orig.topleft)

    # Draw the model-predicted box
    box_rect_model = pygame.Rect(model_pos_x, model_pos_y, box_width, box_height)
    box_surf_model = pygame.Surface((box_width, box_height), pygame.SRCALPHA)
    box_surf_model.fill(GREEN) # Model box is GREEN
    rotated_surf_model = pygame.transform.rotate(box_surf_model, -model_yaw) # model_yaw is in degrees
    rotated_rect_model = rotated_surf_model.get_rect(center=box_rect_model.center)
    screen.blit(rotated_surf_model, rotated_rect_model.topleft)


    # Display joystick values and original drone state
    text_orig_title = font.render("Original Drone (Manual Physics):", True, WHITE)
    screen.blit(text_orig_title, (10, 10))
    text_left_x = font.render(f"LStick X (Yaw In): {current_left_x_orig:.2f}", True, WHITE)
    text_left_y = font.render(f"LStick Y (Z In): {-current_left_y_orig:.2f}", True, WHITE)
    text_right_x = font.render(f"RStick X (X In): {current_right_x_orig:.2f}", True, WHITE)
    text_right_y = font.render(f"RStick Y (Y In): {-current_right_y_orig:.2f}", True, WHITE)
    text_x = font.render(f"Orig X: {pos_x:.2f}", True, WHITE)
    text_y = font.render(f"Orig Y: {pos_y:.2f}", True, WHITE)
    text_z = font.render(f"Orig Z: {pos_z:.2f}", True, WHITE)
    text_yaw_orig = font.render(f"Orig Yaw (rad): {yaw:.2f}", True, WHITE) # Original yaw was in radians
    
    screen.blit(text_left_x, (10, 30))
    screen.blit(text_left_y, (10, 50))
    screen.blit(text_right_x, (10, 70))
    screen.blit(text_right_y, (10, 90))
    screen.blit(text_x, (10, 110))
    screen.blit(text_y, (10, 130))
    screen.blit(text_z, (10, 150))
    screen.blit(text_yaw_orig, (10, 170))

    # Display RC inputs and model-predicted drone state
    text_model_title = font.render("Model Drone (Predicted):", True, WHITE)
    screen.blit(text_model_title, (SCREEN_WIDTH - 250, 10)) # Position on the right
    
    text_rc_ail = font.render(f"RC Ail (RStickX): {rc_aileron:.2f}", True, WHITE)
    text_rc_ele = font.render(f"RC Ele (RStickY): {rc_elevator:.2f}", True, WHITE)
    text_rc_thr = font.render(f"RC Thr (LStickY): {rc_throttle:.2f}", True, WHITE)
    text_rc_rud = font.render(f"RC Rud (LStickX): {rc_rudder:.2f}", True, WHITE)

    text_model_pitch = font.render(f"Mod Pitch: {model_pitch:.2f}", True, WHITE)
    text_model_roll = font.render(f"Mod Roll: {model_roll:.2f}", True, WHITE)
    text_model_yaw = font.render(f"Mod Yaw: {model_yaw:.2f}", True, WHITE)
    text_model_vx = font.render(f"Mod Vx (MPH): {model_xSpeed:.2f}", True, WHITE)
    text_model_vy = font.render(f"Mod Vy (MPH): {model_ySpeed:.2f}", True, WHITE)
    text_model_vz = font.render(f"Mod Vz (MPH): {model_zSpeed:.2f}", True, WHITE)
    text_model_alt = font.render(f"Mod Alt: {model_altitude:.2f}", True, WHITE)
    text_model_x = font.render(f"Mod X: {model_pos_x:.2f}", True, WHITE)
    text_model_y = font.render(f"Mod Y: {model_pos_y:.2f}", True, WHITE)

    y_offset = 30
    screen.blit(text_rc_ail, (SCREEN_WIDTH - 250, y_offset)); y_offset += 20
    screen.blit(text_rc_ele, (SCREEN_WIDTH - 250, y_offset)); y_offset += 20
    screen.blit(text_rc_thr, (SCREEN_WIDTH - 250, y_offset)); y_offset += 20
    screen.blit(text_rc_rud, (SCREEN_WIDTH - 250, y_offset)); y_offset += 20
    y_offset += 10 # spacer
    screen.blit(text_model_pitch, (SCREEN_WIDTH - 250, y_offset)); y_offset += 20
    screen.blit(text_model_roll, (SCREEN_WIDTH - 250, y_offset)); y_offset += 20
    screen.blit(text_model_yaw, (SCREEN_WIDTH - 250, y_offset)); y_offset += 20
    screen.blit(text_model_vx, (SCREEN_WIDTH - 250, y_offset)); y_offset += 20
    screen.blit(text_model_vy, (SCREEN_WIDTH - 250, y_offset)); y_offset += 20
    screen.blit(text_model_vz, (SCREEN_WIDTH - 250, y_offset)); y_offset += 20
    screen.blit(text_model_alt, (SCREEN_WIDTH - 250, y_offset)); y_offset += 20
    screen.blit(text_model_x, (SCREEN_WIDTH - 250, y_offset)); y_offset += 20
    screen.blit(text_model_y, (SCREEN_WIDTH - 250, y_offset)); y_offset += 20


    # Update the full display
    pygame.display.flip()






