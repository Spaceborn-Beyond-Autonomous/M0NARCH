"""
Unitree's pretrained H1 walking policy (unitree_rl_gym, deploy/pre_train/h1/motion.pt), run with NumPy.
No PyTorch needed: the network is an LSTM (64) + MLP (64 -> 32 -> 10), weights read from the file.
"""
import zipfile
import numpy as np


class H1Policy:
    def __init__(self, path):
        z = zipfile.ZipFile(path)
        root = z.namelist()[0].split("/")[0]
        t = lambda k: np.frombuffer(z.read(f"{root}/data/{k}"), dtype=np.float32).copy()
        self.h0, self.c0 = t(0).reshape(64), t(1).reshape(64)
        self.W1, self.b1 = t(2).reshape(32, 64), t(3)
        self.W2, self.b2 = t(4).reshape(10, 32), t(5)
        self.Wih, self.Whh = t(6).reshape(256, 41), t(7).reshape(256, 64)
        self.b = t(8) + t(9)
        self.reset()

    def reset(self):
        self.h, self.c = self.h0.copy(), self.c0.copy()

    def __call__(self, obs):
        g = self.Wih @ obs + self.Whh @ self.h + self.b
        i, f, gg, o = g[:64], g[64:128], g[128:192], g[192:]
        sig = lambda x: 1 / (1 + np.exp(-x))
        self.c = sig(f) * self.c + sig(i) * np.tanh(gg)
        self.h = sig(o) * np.tanh(self.c)
        x = self.W1 @ self.h + self.b1
        x = np.where(x > 0, x, np.expm1(x))                     # ELU
        return self.W2 @ x + self.b2


# settings from unitree_rl_gym deploy/deploy_mujoco/configs/h1.yaml
KPS = np.array([150, 150, 150, 200, 40, 150, 150, 150, 200, 40], np.float32)
KDS = np.array([2, 2, 2, 4, 2, 2, 2, 2, 4, 2], np.float32)
DEFAULT = np.array([0, 0, -0.1, 0.3, -0.2, 0, 0, -0.1, 0.3, -0.2], np.float32)
ANG_VEL_SCALE, DOF_POS_SCALE, DOF_VEL_SCALE, ACTION_SCALE = 0.25, 1.0, 0.05, 0.25
CMD_SCALE = np.array([2.0, 2.0, 0.25], np.float32)
DECIMATION = 10          # policy at 50 Hz, PD at 500 Hz (dt 0.002)
PERIOD = 0.8             # gait phase clock (s)


def gravity_orientation(q):
    qw, qx, qy, qz = q
    return np.array([2 * (-qz * qx + qw * qy), -2 * (qz * qy + qw * qx), 1 - 2 * (qw * qw + qz * qz)])
