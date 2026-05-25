import serial
import struct
import time
import threading
import smbus2
import numpy as np

# ── RPLidar A1 Reader ─────────────────────────────────────────────────────────

class LidarReader:
    """
    Reads RPLidar A1 in a background thread so get_scan() never blocks.
    The LiDAR keeps spinning continuously; get_scan() returns the latest
    completed revolution instantly.
    """
    SYNC_BYTE1 = 0xA5
    SCAN_CMD   = 0x20

    def __init__(self, port='/dev/ttyUSB0', baudrate=115200, timeout=1.0):
        self.port      = port
        self.baudrate  = baudrate
        self.timeout   = timeout
        self.serial    = None
        self._lock     = threading.Lock()
        self._latest   = np.zeros(359, dtype=np.float32)  # 359 to match trained model
        self._running  = False
        self._thread   = None
        self._ready    = False  # True once first full scan is done

    def connect(self):
        self.serial = serial.Serial(self.port, self.baudrate, timeout=self.timeout)
        time.sleep(0.1)
        # Send start scan command and discard 7-byte descriptor
        self.serial.write(bytes([self.SYNC_BYTE1, self.SCAN_CMD]))
        self.serial.read(7)
        # Start background thread
        self._running = True
        self._thread  = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()
        # Wait for first full scan before returning
        print("Waiting for first LiDAR scan...")
        while not self._ready:
            time.sleep(0.05)
        print("LiDAR ready.")

    def _read_loop(self):
        current_scan = {}
        while self._running:
            try:
                raw = self.serial.read(5)
                if len(raw) < 5:
                    continue

                b0, b1, b2, b3, b4 = raw

                # Validate sync bits
                if (b0 & 0x01) == ((b0 >> 1) & 0x01):
                    continue  # bad packet

                quality  = b0 >> 2
                angle    = ((b1 >> 1) | (b2 << 7)) / 64.0
                distance = (b3 | (b4 << 8)) / 4000.0

                # New revolution starts — publish completed scan
                if (b0 & 0x01) == 1 and current_scan:
                    result = np.zeros(359, dtype=np.float32)
                    for deg, dist in current_scan.items():
                        if deg < 359:
                            result[deg] = dist
                    with self._lock:
                        self._latest = result
                    self._ready = True
                    current_scan = {}

                angle_int = int(angle) % 360
                if quality > 0 and distance > 0:
                    current_scan[angle_int] = distance

            except Exception:
                continue

    def get_scan(self):
        """Returns the latest 359-element scan instantly — never blocks."""
        with self._lock:
            return self._latest.copy()

    def disconnect(self):
        self._running = False
        if self.serial and self.serial.is_open:
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