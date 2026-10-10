#!/usr/bin/env python3
"""Print the first MUJOCO_EGL_DEVICE_ID that can render a MuJoCo frame (empty line if none).

MuJoCo's EGL backend (no MUJOCO_EGL_DEVICE_ID) walks every EGL device and calls eglInitialize on
each; on a node with a faulty GPU the NVIDIA driver aborts the process (SIGABRT) instead of
failing that call, so the rollout dies before rendering. Probing each device in a subprocess
sidesteps the abort. Run on the host venv (mujoco + PyOpenGL) with the GPU job's cgroup.
"""
import os
import subprocess
import sys

_PROBE = (
    "import mujoco;m=mujoco.MjModel.from_xml_string('<mujoco><worldbody><geom size=\"1\"/></worldbody></mujoco>');"
    "r=mujoco.Renderer(m,16,16);r.update_scene(mujoco.MjData(m));r.render();print('ok')"
)


def _device_count() -> int:
    try:
        os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
        from OpenGL import EGL
        from OpenGL.EGL.EXT.device_enumeration import eglQueryDevicesEXT
        n = EGL.EGLint()
        eglQueryDevicesEXT(0, None, n)
        return int(n.value)
    except Exception:  # noqa: BLE001 - any enumeration failure -> bounded scan
        return 16


def main() -> int:
    for i in range(_device_count()):
        env = dict(os.environ, MUJOCO_GL="egl", MUJOCO_EGL_DEVICE_ID=str(i))
        try:
            r = subprocess.run([sys.executable, "-c", _PROBE], env=env, capture_output=True, text=True, timeout=90)
        except subprocess.TimeoutExpired:
            continue
        if r.returncode == 0 and "ok" in r.stdout:
            print(i)
            return 0
    print("")
    return 1


if __name__ == "__main__":
    sys.exit(main())
