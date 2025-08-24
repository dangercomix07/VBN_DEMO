import time
import cv2
import numpy as np
from capture_frame import FrameSource
from led_detection import detect_leds
from pose_estimation import identify_and_order, estimate_pose
from display import draw_raw_points, draw_pattern, draw_reprojections, draw_hud

# ---------- Pattern geometry (meters) ----------
R = 0.020  # radius from center to each of the 4 outer LEDs

# ---------- Camera intrinsics & distortion ----------
camMatrix = np.array([[2593,    0, 1614],
                      [   0, 2588, 1213],
                      [   0,    0,    1]], dtype=np.float64)

distCoeff = np.array([ 2.18984921e-01, -5.80493965e-01, 1.15200278e-04,
                      -2.04177566e-03, 4.48611005e-01], dtype=np.float64)

def _los_from_center(center_uv, K):
    """Compute LOS azimuth/elevation (deg) from pattern center and intrinsics."""
    fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
    u, v = float(center_uv[0]), float(center_uv[1])
    x_n = (u - cx) / fx
    y_n = (v - cy) / fy
    az  = np.degrees(np.arctan2(x_n, 1.0))    # Az (ψ)
    el  = np.degrees(np.arctan2(-y_n, 1.0))   # El (θ)
    return az, el

def main():
    cap = FrameSource()
    t0 = time.time(); frames = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            print("No frame from camera. Edit SOURCE in capture_frame.py.")
            break

        frames += 1
        fps = frames / max(1e-6, (time.time() - t0))

        # --- detect up to 5 bright blobs ---
        pts = detect_leds(frame, max_pts=5)
        vis = frame.copy()
        draw_raw_points(vis, pts)  # cyan dots for all detections

        hud = []
        if len(pts) >= 5:
            res = identify_and_order(pts)
            if res["ok"]:
                C        = res["center_uv"]
                off      = res["offset_uv"]
                outs     = res["outer_uv"]
                ordered  = res["ordered_pts2d"]

                draw_pattern(vis, C, off, outs)

                # --- hybrid pose: analytic + PnP ---
                try:
                    pose = estimate_pose(ordered, camMatrix, distCoeff, R)
                except Exception as e:
                    pose = {"analytic": {"ok": False, "reason": str(e)},
                            "pnp":      {"ok": False, "reason": "exception"}}

                # ---- Analytic: RPY + LOS + Range ----
                az_a, el_a = _los_from_center(C, camMatrix)
                ap = pose.get("analytic", {})
                if ap.get("ok", False):
                    φa, θa, ψa = ap["rpy321_deg"]
                    Ra = ap["range_m"]
                    hud += [
                        "Analytic:",
                        f"R{φa:6.1f}deg  P{θa:6.1f}deg  Y{ψa:6.1f}deg",
                        f"LOS: Az{az_a:6.1f}deg  El{el_a:6.1f}deg   R{Ra:6.3f} m",
                    ]
                else:
                    hud += ["Analytic: unavailable"]

                # ---- PnP: RPY + LOS + Range ----
                pnp = pose.get("pnp", {"ok": False})
                if pnp.get("ok", False):
                    φp, θp, ψp = pnp["rpy321_deg"]
                    Rp = pnp["range_m"]
                    az_p, el_p = _los_from_center(C, camMatrix)  # reuse center-based LOS
                    hud += [
                        "PnP:",
                        f"R{φp:6.1f}deg  P{θp:6.1f}deg  Y{ψp:6.1f}deg",
                        f"LOS: Az{az_p:6.1f}deg  El{el_p:6.1f}deg R{Rp:6.3f} m",
                    ]
                    # If your pose_estimation adds reproj/z_cam later, this will draw them safely:
                    draw_reprojections(vis, pnp.get("reproj", None), pnp.get("z_cam", None))
                else:
                    hud += [f"PnP: {pnp.get('reason', pnp.get('error', 'unavailable'))}"]
            else:
                hud += [res.get("reason", "pattern id failed")]
        else:
            hud += [f"Need 5 LEDs (got {len(pts)})"]

        draw_hud(vis, hud, fps=fps)

        cv2.imshow("VBN | RPY + LOS + Range", vis)

        # PRINTING TO CONSOLE
        print(f"\rFPS: {fps:5.1f}", end="")
        print("LEDS detected:", len(pts), end="; ")
        if len(pts) >= 5:
            print("Pattern OK.", end="; ")
            print("LEDs Postion (px):", np.array2string(pts, precision=1, floatmode='fixed'), end="; ")

        if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
            break

        if cv2.waitKey(1) & 0xFF in (ord('s'), ord(' ')):
            cv2.imwrite(f"vbn_{int(time.time())}.png", vis)
            print(" [saved]", end="")   

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
