# capture_frame.py
# Tiny, cross-platform capture:
# - On Raspberry Pi: uses Picamera2 (libcamera) with short exposure
# - Elsewhere: uses OpenCV VideoCapture with best-effort low exposure
# Frames are returned as BGR (OpenCV-ready).

import cv2, platform
import numpy as np

# ===== minimal knobs =====
SOURCE        = 0          # OpenCV device index or "rtsp://..." URL (ignored by Picamera2)
WIDTH, HEIGHT = 1280, 720  # set to None to keep defaults
FPS           = 30
EXPOSURE_MS   = 5.0        # ~2–10 ms works well for LEDs
USE_PICAMERA2 = True       # set False to force OpenCV even on a Pi
# =========================

class FrameSource:
    def __init__(self):
        self.picam2 = None
        self.cap = None

        # Try Picamera2 on Linux (i.e., Raspberry Pi). Falls back to OpenCV if not available.
        if USE_PICAMERA2 and platform.system() == "Linux" and isinstance(SOURCE, int):
            try:
                from picamera2 import Picamera2
                self.picam2 = Picamera2()
                size = (int(WIDTH), int(HEIGHT)) if (WIDTH and HEIGHT) else (1280, 720)
                cfg = self.picam2.create_video_configuration(main={"size": size, "format": "RGB888"})
                self.picam2.configure(cfg)

                # lock exposure/gain/awb for stable LED detection
                self.picam2.set_controls({
                    "AeEnable": False,
                    "ExposureTime": int(max(1, EXPOSURE_MS * 1000.0)),  # microseconds
                    "AnalogueGain": 1.0,
                    "AwbEnable": False,
                })
                if FPS:
                    frame_us = int(1_000_000 / max(1, int(FPS)))
                    try:
                        self.picam2.set_controls({"FrameDurationLimits": (frame_us, frame_us)})
                    except Exception:
                        pass
                self.picam2.start()
                return
            except Exception as e:
                print(f"[capture] Picamera2 not available, using OpenCV instead. ({e})")

        # OpenCV fallback (PC or Pi without Picamera2)
        self.cap = cv2.VideoCapture(SOURCE)
        if WIDTH:  self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  int(WIDTH))
        if HEIGHT: self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(HEIGHT))
        if FPS:    self.cap.set(cv2.CAP_PROP_FPS,          int(FPS))

        # best-effort short exposure + low gain (many cams ignore; harmless if so)
        try:
            self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # manual (common on V4L2)
            self.cap.set(cv2.CAP_PROP_EXPOSURE, -7)         # dshow-ish value; try tweaking
            self.cap.set(cv2.CAP_PROP_GAIN, 0)
            self.cap.set(cv2.CAP_PROP_AUTO_WB, 0)
        except Exception:
            pass

    def read(self):
        if self.picam2 is not None:
            arr = self.picam2.capture_array("main")  # RGB
            if arr is None or arr.size == 0:
                return False, None
            return True, cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        else:
            return self.cap.read()

    def release(self):
        if self.picam2 is not None:
            try: self.picam2.stop()
            except: pass
        if self.cap is not None:
            self.cap.release()

# Optional generator if you like for-loops:
def capture_frame():
    src = FrameSource()
    try:
        while True:
            ok, frame = src.read()
            if not ok: break
            yield frame
    finally:
        src.release()
