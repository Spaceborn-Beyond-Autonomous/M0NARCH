"""
M0NARCH Pick & Place vision library: the robot's head camera and OpenCV detection.

  add_eye_camera(spec)          camera "eye_cam" on the H1's head, looking ahead and 62 deg down
                                (the Pick & Place scene had none, so the video thread crashed)
  EyeCamera(m)                  renders that camera: colour image + depth image
  world_image(...)              pinhole back-projection of every pixel (depth) -> 3D world point
  detect_blobs(bgr, ranges)     OpenCV: HSV colour mask -> open/close clean-up -> contours

Used by pnp_sorting.py. Runs in the control thread (like the Obstacle Avoidance camera).
"""
import math

import cv2
import mujoco
import numpy as np

EYE_CAMERA = "eye_cam"
EYE_BODY = "torso_link"
EYE_POS = (0.13, 0.0, 0.62)          # m, in the torso frame (front of the head)
EYE_PITCH_DEG = 62.0                 # looking down at the table
EYE_FOVY_DEG = 55.0
IMAGE_SIZE = (640, 480)              # width, height

# HSV ranges (OpenCV hue 0-180)
COLOUR_RANGES = {
    "red": (((0, 120, 70), (10, 255, 255)), ((170, 120, 70), (180, 255, 255))),
    "green": (((40, 120, 70), (85, 255, 255)),),
    "blue": (((100, 150, 60), (130, 255, 255)),),
}
MIN_AREA_PX = 60                     # smaller blobs are noise


def add_eye_camera(spec):
    """Add the head camera to a model spec (before spec.compile())."""
    if any(c.name == EYE_CAMERA for c in spec.cameras):
        return
    body = spec.body(EYE_BODY)
    if body is None:
        return
    s, c = math.sin(math.radians(EYE_PITCH_DEG)), math.cos(math.radians(EYE_PITCH_DEG))
    # camera axes in the torso frame: x = right (-y), y = up tilted forward, looks along -z
    rot = np.array([0, s, -c, -1, 0, 0, 0, c, s], float)
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, rot)
    cam = body.add_camera(name=EYE_CAMERA, pos=list(EYE_POS), fovy=EYE_FOVY_DEG)
    cam.quat = quat


class EyeCamera:
    """Colour + depth images from the head camera (create and use in the same thread)."""

    def __init__(self, m):
        self.cam = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, EYE_CAMERA)
        self.W = min(IMAGE_SIZE[0], m.vis.global_.offwidth)
        self.H = min(IMAGE_SIZE[1], m.vis.global_.offheight)
        self.renderer = mujoco.Renderer(m, self.H, self.W)

    def capture(self, d):
        r = self.renderer
        r.update_scene(d, camera=self.cam)
        rgb = r.render().copy()
        r.enable_depth_rendering()
        r.update_scene(d, camera=self.cam)
        depth = r.render().copy()
        r.disable_depth_rendering()
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), depth

    def close(self):
        self.renderer.close()


def world_image(m, d, cam, depth):
    """Pinhole back-projection of every pixel -> world points (H x W x 3); NaN where depth is invalid."""
    H, W = depth.shape
    fy = 0.5 * H / math.tan(math.radians(m.cam_fovy[cam]) / 2)
    u, v = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
    z = depth.astype(float)
    pts_c = np.stack([(u - W / 2) / fy * z, -(v - H / 2) / fy * z, -z], -1)   # x right, y up, looks along -z
    R = d.cam_xmat[cam].reshape(3, 3)
    pts = pts_c @ R.T + d.cam_xpos[cam]
    pts[(z < 0.05) | (z > 5.0)] = np.nan
    return pts


def detect_blobs(bgr, ranges, keep=None):
    """OpenCV: every blob of one colour, largest first. Each: dict(mask, box, centre, area).
    keep: optional 0/255 mask of pixels allowed (e.g. only the table top, not the floor)."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = np.zeros(hsv.shape[:2], np.uint8)
    for lo, hi in ranges:
        mask |= cv2.inRange(hsv, np.array(lo), np.array(hi))
    if keep is not None:
        mask &= keep
    k = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    blobs = []
    for c in sorted(contours, key=cv2.contourArea, reverse=True):
        area = cv2.contourArea(c)
        if area < MIN_AREA_PX:
            break
        (cx, cy), (w, h), ang = cv2.minAreaRect(c)
        only = np.zeros_like(mask)
        cv2.drawContours(only, [c], -1, 255, -1)
        blobs.append(dict(mask=only & mask, box=cv2.boxPoints(((cx, cy), (w, h), ang)).astype(int),
                          centre=(int(cx), int(cy)), area=area))
    return blobs
