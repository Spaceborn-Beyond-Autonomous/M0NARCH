"""
Checks the REAL server.py after apply_physics_walk.py: Stand, Walk and Walk Avoid all run
under physics in the server's own control loop (render thread skipped, simulated time).

  python test_server_physics.py            # about 1.5-2 minutes
"""
import os
import sys
import threading
import time

os.environ.setdefault("MUJOCO_GL", "egl")
_start = threading.Thread.start


def _no_render(self):
    if getattr(self, "_target", None) is not None and self._target.__name__ == "render_worker":
        return
    _start(self)


threading.Thread.start = _no_render
sys.argv = [sys.argv[0]]
import server  # noqa: E402  (starts the control thread only)

if server.walker is None:
    sys.exit("physics walking is OFF in server.py (policy missing?)")


def run(mode, d, seconds, check):
    server.execute_command(mode)
    t0 = None
    while True:
        with server.sim.lock:
            t, z, x, y = float(d.time), float(d.qpos[2]), float(d.qpos[0]), float(d.qpos[1])
        t0 = t if t0 is None else t0
        if z < 0.6 or t - t0 >= seconds or check():
            return t - t0, z, x, y
        time.sleep(0.05)


results = []
t, z, x, y = run("stand", server.data_normal, 10, lambda: False)
results.append(("STAND 10 s", z > 0.8, f"pelvis z={z:.2f} m"))
t, z, x, y = run("walk", server.data_normal, 10, lambda: False)
results.append(("WALK 10 s (physics)", z > 0.8 and x > 2.0, f"walked x={x:.2f} m, z={z:.2f} m"))
t, z, x, y = run("walk_avoid", server.data_obs, 90, lambda: server.nav.status == "goal reached")
results.append(("WALK AVOID to goal", z > 0.8 and server.nav.status == "goal reached",
                f"{server.nav.status} after {t:.0f} s at ({x:.1f}, {y:.1f}), goal {server.NAV_GOAL}"))
server.sim.running = False
print("\nserver.py physics check")
for name, ok, info in results:
    print(f"  {name:22s} {'PASS' if ok else 'FAIL'}   {info}")
os._exit(0 if all(r[1] for r in results) else 1)
