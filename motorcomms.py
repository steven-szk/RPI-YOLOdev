import serial # type: ignore
import time
import math

SLIP_SKID_FACTOR = 1.2

class UnibotDrive:
    def __init__(self, port_fl='/dev/ttyAMA2', port_fr='/dev/ttyAMA4', 
                 port_bl='/dev/ttyAMA0', port_br='/dev/ttyAMA3'):
        """
        
        Initializes the 4 separate UART ports for the motors.
        Assumes factory default motor ID = 1 for all motors, since they 
        are on independent data lines.
        """
        print("Initializing Unibot motors...")
        opened = []
        try:
            self.motor_fl = serial.Serial(port_fl, 115200, timeout=1) # right dir
            opened.append(self.motor_fl)
            self.motor_fr = serial.Serial(port_fr, 115200, timeout=1) # right dir
            opened.append(self.motor_fr)
            self.motor_bl = serial.Serial(port_bl, 115200, timeout=1) # 
            opened.append(self.motor_bl)
            self.motor_br = serial.Serial(port_br, 115200, timeout=1) # wrong dir
            opened.append(self.motor_br)
            
            self.wheel_diameter_mm = 74.5
            self.track_width_mm = 180.0
            
            # Group them in a list for easy iteration when applying commands to all
            self.all_motors = [self.motor_fl, self.motor_fr, self.motor_bl, self.motor_br]
            print("Ready!")
            
        except serial.SerialException as e:
            print(f"CRITICAL ERROR: Could not open serial ports. {e}")
            for s in opened:                 # don't leak the ports we did open
                s.close()
            raise

    def _calc_crc8(self, data: bytearray) -> int:
        """Internal helper to calculate the DDSM210 checksum."""
        crc = 0x00
        for byte in data:
            crc ^= byte
            for _ in range(8):
                if crc & 0x01:
                    crc = (crc >> 1) ^ 0x8C
                else:
                    crc >>= 1
        return crc

    def _send_velocity(self, ser, rpm, accel=20):
        """
        Internal helper to format and send a velocity command to a specific port.
        accel: Acceleration time per 1 RPM in 0.1ms increments (0-255). 
               Default 20 = 2ms per 1 RPM change.
        """
        rpm = max(-210.0, min(210.0, rpm))
        speed_val = int(rpm * 10)
        
        if speed_val < 0:
            speed_val = (1 << 16) + speed_val
            
        high_byte = (speed_val >> 8) & 0xFF
        low_byte = speed_val & 0xFF
        
        # Clamp acceleration to a valid byte (0-255)
        accel_byte = max(0, min(255, int(accel)))
        
        # Mode 0x02 (Velocity)
        mode_cmd = bytearray([0x01, 0xA0, 0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00])
        mode_cmd.append(self._calc_crc8(mode_cmd))
        
        # Velocity command (Byte 6 is Acceleration)
        vel_cmd = bytearray([0x01, 0x64, high_byte, low_byte, 0x00, 0x00, accel_byte, 0x00, 0x00])
        vel_cmd.append(self._calc_crc8(vel_cmd))
        
        ser.write(mode_cmd)
        time.sleep(0.01) # Tiny delay for mode switch
        ser.write(vel_cmd)

    def _send_brake(self, ser):
        """Internal helper to format and send a brake command to a specific port."""
        mode_cmd = bytearray([0x01, 0xA0, 0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00])
        mode_cmd.append(self._calc_crc8(mode_cmd))
        
        brake_cmd = bytearray([0x01, 0x64, 0x00, 0x00, 0x00, 0x00, 0x00, 0xFF, 0x00])
        brake_cmd.append(self._calc_crc8(brake_cmd))
        
        ser.write(mode_cmd)
        time.sleep(0.01)
        ser.write(brake_cmd)

    def _get_total_angle(self, ser):
        """
        Queries the motor for its absolute position.
        Uses robust software unwrapping to track continuous laps.
        """
        cmd = bytearray([0x01, 0x74, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00])
        cmd.append(self._calc_crc8(cmd))
        
        # Clear out any stale data before asking for a fresh position
        ser.reset_input_buffer()
        ser.write(cmd)
        
        start_t = time.time()
        while ser.in_waiting < 10:
            if time.time() - start_t > 0.05:
                return None
                
        resp = ser.read(10)
        
        if len(resp) == 10 and self._calc_crc8(bytearray(resp[:9])) == resp[9]:
            # The absolute position is strictly 0 to 32767
            pos_raw = int.from_bytes(resp[6:8], byteorder='big', signed=False)
            
            # Convert raw position to 0-360 degrees
            pos_degrees = (pos_raw / 32767.0) * 360.0 
            
            # --- Software Lap Unwrapping Logic ---
            # Initialize dictionaries to track laps for each unique motor port
            if not hasattr(self, '_sw_laps'):
                self._sw_laps = {}
                self._last_pos_deg = {}
                
            port_id = id(ser)
            if port_id not in self._sw_laps:
                self._sw_laps[port_id] = 0
                self._last_pos_deg[port_id] = pos_degrees
                
            delta = pos_degrees - self._last_pos_deg[port_id]
            
            # If the wheel jumps more than 180° backward, it wrapped forward a lap
            if delta < -180.0:
                self._sw_laps[port_id] += 1
            # If the wheel jumps more than 180° forward, it wrapped backward a lap
            elif delta > 180.0:
                self._sw_laps[port_id] -= 1
                
            self._last_pos_deg[port_id] = pos_degrees
            
            # Calculate infinite continuous degrees
            total_continuous_degrees = (self._sw_laps[port_id] * 360.0) + pos_degrees
            return total_continuous_degrees
            
        return None

    # ---------------------------------------------------------
    # PUBLIC MOVEMENT COMMANDS
    # ---------------------------------------------------------

    def set_velocity(self, forward_rpm, turn_rpm=0.0, accel=2):
        """Non-blocking: set continuous forward + turn speed, then return at once.

        Positive forward_rpm drives forward; positive turn_rpm turns right.
        The two are mixed into differential wheel speeds, so the caller can keep
        its perception loop running while the robot moves (unlike drive_distance/
        turn_angle, which block until the move completes).
        """
        forward_rpm = forward_rpm - 0.2*abs(turn_rpm)
        left = forward_rpm + turn_rpm
        right = forward_rpm - turn_rpm
        # Left side moves normally; right side is physically inverted, so negate.
        self._send_velocity(self.motor_fl, left, accel)
        self._send_velocity(self.motor_bl, left, accel)
        self._send_velocity(self.motor_fr, -right, accel)
        self._send_velocity(self.motor_br, -right, accel)

    def stop(self):
        """Engages the electronic brake on all four wheels immediately."""
        #print("Engaging brakes!")
        for motor in self.all_motors:
            self._send_brake(motor)


    def drive_distance(self, speed_rpm, distance_cm=0.0, accel=2, stop_condition=None):
        """
        Drives the robot straight for a set distance in centimeters.
        Uses absolute encoder feedback for precision with a fast-then-slow glide.
        stop_condition: An optional callable that returns True if the drive should abort early.
        """
        if distance_cm == 0:
            return

        # Determine direction: distance and speed signs dictate forward/backward
        direction = 1 if (speed_rpm * distance_cm) >= 0 else -1
        
        target_rpm = abs(speed_rpm)
        slow_rpm = 20.0 
        
        # Calibration offsets (You may need to tune these based on robot weight)
        brake_offset_degrees = 15.0  # Degrees before target to cut power and brake
        slowdown_degrees = 90.0      # Degrees before target to decelerate to slow_rpm

        # 1. Target calculation
        circumference_cm = (math.pi * self.wheel_diameter_mm) / 10.0
        target_wheel_rotation = (abs(distance_cm) / circumference_cm) * 360.0
        
        # 2. Get starting position from the front-left encoder
        start_angle = self._get_total_angle(self.motor_fl)
        if start_angle is None:
            print("Error: Could not read encoder. Aborting drive.")
            return

        dir_str = "Forward" if direction > 0 else "Backward"
        print(f"Driving {dir_str} {abs(distance_cm)} cm...")

        # Dynamic timeout: Base 10 seconds + extra time for longer distances
        timeout_duration = 10.0 + (abs(distance_cm) * 0.2)
        timeout_start = time.time()
        current_rpm = target_rpm
        
        # Start moving fast (Note: Right side is physically inverted)
        self._send_velocity(self.motor_fl, current_rpm * direction, accel)
        self._send_velocity(self.motor_fr, -current_rpm * direction, accel)
        self._send_velocity(self.motor_bl, current_rpm * direction, accel)
        self._send_velocity(self.motor_br, -current_rpm * direction, accel)

        last_condition_check = time.time()

        while True:
            # Check the stopping condition periodically (e.g., every 0.15 seconds)
            if stop_condition is not None and (time.time() - last_condition_check > 0.4):
                if stop_condition():
                    print("\nStop condition met! Aborting drive early.")
                    break
                last_condition_check = time.time()

            current_angle = self._get_total_angle(self.motor_fl)
            
            if current_angle is not None:
                distance_moved_deg = abs(current_angle - start_angle)
                degrees_remaining = target_wheel_rotation - distance_moved_deg

                # Exit condition: Hit the brakes slightly early to glide to a stop
                if degrees_remaining <= brake_offset_degrees:
                    break 

                # Speed control: Trigger the hardware deceleration ramp to slow_rpm
                if degrees_remaining <= slowdown_degrees and current_rpm != slow_rpm:
                    current_rpm = slow_rpm
                    self._send_velocity(self.motor_fl, current_rpm * direction, accel)
                    self._send_velocity(self.motor_fr, -current_rpm * direction, accel)
                    self._send_velocity(self.motor_bl, current_rpm * direction, accel)
                    self._send_velocity(self.motor_br, -current_rpm * direction, accel)

            # Let the serial bus breathe
            time.sleep(0.01)

            # Safety timeout
            if time.time() - timeout_start > timeout_duration:
                print("\nTimeout: Drive took too long.")
                break

        # Slam the brakes
        self.stop()
        print() # Move to new line
        
        # Settle physically before reading the final position
        time.sleep(0.2) 
        final_angle = self._get_total_angle(self.motor_fl)
        if final_angle is not None:
            final_distance_deg = abs(final_angle - start_angle)
            final_distance_cm = (final_distance_deg / 360.0) * circumference_cm
            print(f"Drive complete. Final Distance: {final_distance_cm:.2f} cm")


    def turn_angle(self, angle_degrees, rpm=60, accel=20, skid_factor=SLIP_SKID_FACTOR):
        """
        Turns the robot using a fast-then-slow approach.
        Adjusted to account for hardware deceleration gliding.
        """
        if angle_degrees == 0:
            return

        abs_angle = abs(angle_degrees)
        rpm = abs(rpm)
        slow_rpm = 20.0 
        
        # --- Dynamic Parameter Tuning for Small vs. Large Angles ---
        if abs_angle > 20.0:
            # For larger turns, use the fast-then-slow approach
            current_rpm = rpm
            slowdown_degrees = 20.0
            # Brake offset to account for glide from slow_rpm (comment said 2.5, 10 is too high)
            brake_offset_degrees = 2.5 
        elif abs_angle > 3.0:
            # For medium turns, start slow but use a moderate brake offset
            current_rpm = slow_rpm
            slowdown_degrees = 0 # No slowdown phase needed
            brake_offset_degrees = 1.5
        else:
            # For very small, precise turns (<3 deg), start slow and use a minimal offset
            current_rpm = slow_rpm
            slowdown_degrees = 0 # No slowdown phase needed
            # This needs to be less than the smallest angle you want to turn (e.g. < 1.0)
            brake_offset_degrees = 0.5

        # 1. Target calculation
        ratio = self.track_width_mm / self.wheel_diameter_mm
        target_wheel_rotation = abs_angle * ratio * skid_factor
        
        # 2. Get starting position
        start_angle = self._get_total_angle(self.motor_fl)
        if start_angle is None:
            print("Error: Could not read encoder. Aborting turn.")
            return

        direction_str = "Right" if angle_degrees > 0 else "Left"
        print(f"Turning {direction_str} {abs_angle}°...")

        timeout_start = time.time()
        
        # Apply physical turn direction
        turn_dir = 1 if angle_degrees > 0 else -1
        
        # Start moving fast with hardware acceleration
        self._send_velocity(self.motor_fl, current_rpm * turn_dir, accel)
        self._send_velocity(self.motor_bl, current_rpm * turn_dir, accel)
        self._send_velocity(self.motor_fr, current_rpm * turn_dir, accel)
        self._send_velocity(self.motor_br, current_rpm * turn_dir, accel)

        while True:
            current_angle = self._get_total_angle(self.motor_fl)
            
            if current_angle is not None:
                distance_moved = abs(current_angle - start_angle)
                
                # Convert wheel distance back into the robot's current turning angle
                robot_current_angle = distance_moved / (ratio * skid_factor)
                degrees_remaining = abs_angle - robot_current_angle

                # Exit condition: Hit the brakes slightly early to glide to a stop
                if degrees_remaining <= brake_offset_degrees:
                    break 

                # Speed control: Trigger the hardware deceleration ramp to slow_rpm
                if current_rpm != slow_rpm and degrees_remaining <= slowdown_degrees:
                    current_rpm = slow_rpm
                    self._send_velocity(self.motor_fl, current_rpm * turn_dir, accel)
                    self._send_velocity(self.motor_bl, current_rpm * turn_dir, accel)
                    self._send_velocity(self.motor_fr, current_rpm * turn_dir, accel)
                    self._send_velocity(self.motor_br, current_rpm * turn_dir, accel)

            # Let the serial bus breathe
            time.sleep(0.01)

            # Safety timeout (10 seconds)
            if time.time() - timeout_start > 10.0:
                print("\nTimeout: Turn took too long.")
                break

        # Slam the brakes
        self.set_velocity(forward_rpm=0)
        print() # Move to new line so we don't overwrite any potential progress bars
        
        # Give it a tiny moment to settle physically before reading final angle
        time.sleep(0.2) # Increased settle time slightly due to the rolling stop
        final_angle = self._get_total_angle(self.motor_fl)
        if final_angle is not None:
            final_distance = abs(final_angle - start_angle)
            final_robot_angle = final_distance / (ratio * skid_factor)
            #print(f"Turn complete. Final Angle: {final_robot_angle:.2f}°")
            
    
    def close(self):
        """Safely closes all serial ports. Call this when shutting down."""
        self.stop() # Brake before shutting down connections
        time.sleep(0.1)
        for motor in self.all_motors:
            if motor.is_open:
                motor.close()
        print("All motor ports closed safely.")
