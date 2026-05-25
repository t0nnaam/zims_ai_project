import serial
import struct
import time
import smbus2
import numpy as np

# ── RPLidar A1 Reader ─────────────────────────────────────────────────────────

class LidarReader:
    """
    Reads 360-degree scan data directly from RPLidar A1 over serial.
    Returns a 360-element array (one value per degree, in meters).
    """
    SYNC_BYTE1 = 0xA5
    SYNC_BYTE2 = 0x5A
    SCAN_CMD   = 0x20

    def __init__(self, port='/dev/ttyUSB0', baudrate=115200, timeout=1.0):
        self.port     = port
        self.baudrate = baudrate
        self.timeout  = timeout
        self.serial   = None
        self._scan_buffer = {}  # angle_int -> distance_m

    def connect(self):
        self.serial = serial.Serial(self.port, self.baudrate, timeout=self.timeout)
        time.sleep(0.1)
        self._start_scan()

    def _start_scan(self):
        # Send start scan command
        self.serial.write(bytes([self.SYNC_BYTE1, self.SCAN_CMD]))
        # Read and discard the response descriptor (7 bytes)
        self.serial.read(7)

    def _read_packet(self):
        """Read one 5-byte scan packet from the LiDAR."""
        raw = self.serial.read(5)
        if len(raw) < 5:
            return None

        b0, b1, b2, b3, b4 = raw

        # Validate sync bits
        start_bit     = b0 & 0x01
        inv_start_bit = (b0 >> 1) & 0x01
        if start_bit == inv_start_bit:
            return None  # bad packet, skip

        quality  = b0 >> 2
        angle    = ((b1 >> 1) | (b2 << 7)) / 64.0      # degrees
        distance = (b3 | (b4 << 8)) / 4000.0            # meters (raw is in mm*4)

        return angle, distance, quality

    def get_scan(self):
        """
        Collect one full 360° scan.
        Blocks until a complete revolution is detected.
        Returns a numpy array of shape (360,) in meters.
        Unreachable/invalid points are set to 0.0.
        """
        scan = {}
        first_angle_seen = False
        start_angle = None

        while True:
            packet = self._read_packet()
            if packet is None:
                continue

            angle, distance, quality = packet

            if not first_angle_seen:
                start_angle      = angle
                first_angle_seen = True

            angle_int = int(angle) % 360
            if quality > 0 and distance > 0:
                scan[angle_int] = distance

            # Detect when we've gone past 355° — full revolution done
            if first_angle_seen and angle > 355:
                break

        # Build fixed-size 360 array
        result = np.zeros(360, dtype=np.float32)
        for deg, dist in scan.items():
            result[deg] = dist

        return result

    def disconnect(self):
        if self.serial and self.serial.is_open:
            # Send stop command
            self.serial.write(bytes([self.SYNC_BYTE1, 0x25]))
            time.sleep(0.1)
            self.serial.close()


# ── GY-521 (MPU6050) IMU Reader ───────────────────────────────────────────────

class IMUReader:
    """
    Reads 6-axis data from GY-521 (MPU6050) over I2C.
    Returns [accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z] in SI units.
      - Accelerometer: m/s²
      - Gyroscope:     deg/s
    """
    MPU6050_ADDR       = 0x68
    PWR_MGMT_1         = 0x6B
    ACCEL_XOUT_H       = 0x3B
    GYRO_XOUT_H        = 0x43
    ACCEL_SCALE_FACTOR = 16384.0   # ±2g range  -> LSB/g
    GYRO_SCALE_FACTOR  = 131.0     # ±250°/s range -> LSB/(°/s)
    G_TO_MS2           = 9.80665

    def __init__(self, bus=1, address=0x68):
        self.bus     = smbus2.SMBus(bus)
        self.address = address
        self._wake()

    def _wake(self):
        # Clear sleep bit to wake the MPU6050
        self.bus.write_byte_data(self.address, self.PWR_MGMT_1, 0x00)
        time.sleep(0.1)

    def _read_word_signed(self, reg):
        high = self.bus.read_byte_data(self.address, reg)
        low  = self.bus.read_byte_data(self.address, reg + 1)
        val  = (high << 8) | low
        if val >= 0x8000:
            val -= 0x10000
        return val

    def get_reading(self):
        """
        Returns [accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z]
          - accel in m/s²
          - gyro  in deg/s
        """
        ax_raw = self._read_word_signed(self.ACCEL_XOUT_H)
        ay_raw = self._read_word_signed(self.ACCEL_XOUT_H + 2)
        az_raw = self._read_word_signed(self.ACCEL_XOUT_H + 4)

        gx_raw = self._read_word_signed(self.GYRO_XOUT_H)
        gy_raw = self._read_word_signed(self.GYRO_XOUT_H + 2)
        gz_raw = self._read_word_signed(self.GYRO_XOUT_H + 4)

        ax = (ax_raw / self.ACCEL_SCALE_FACTOR) * self.G_TO_MS2
        ay = (ay_raw / self.ACCEL_SCALE_FACTOR) * self.G_TO_MS2
        az = (az_raw / self.ACCEL_SCALE_FACTOR) * self.G_TO_MS2

        gx = gx_raw / self.GYRO_SCALE_FACTOR
        gy = gy_raw / self.GYRO_SCALE_FACTOR
        gz = gz_raw / self.GYRO_SCALE_FACTOR

        return [ax, ay, az, gx, gy, gz]

    def close(self):
        self.bus.close()


# ── Quick test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Testing LiDAR...")
    lidar = LidarReader()
    lidar.connect()
    scan = lidar.get_scan()
    print(f"LiDAR OK — got {np.count_nonzero(scan)} valid points")
    print(f"Nearest object: {scan[scan > 0].min():.2f}m" if scan.any() else "No objects detected")
    lidar.disconnect()

    print("\nTesting IMU...")
    imu = IMUReader()
    reading = imu.get_reading()
    print(f"IMU OK — accel: {reading[:3]}, gyro: {reading[3:]}")
    imu.close()