#!/usr/bin/env python3
import json, time, threading, traceback, signal, sys
import zmq

import rclpy
from rclpy.executors import MultiThreadedExecutor
from controller import KinovaGen3Controller  # your node class on A


def build_dispatch(ctrl):
    return {
        "joint_movement": lambda args, kwargs: ctrl.joint_movement(*args, **kwargs),
        "cartesian_movement": lambda args, kwargs: ctrl.cartesian_movement(*args, **kwargs),
        "publish_cartesian_velocity": lambda args, kwargs: ctrl.publish_cartesian_velocity(*args, **kwargs),
        "gripper_movement": lambda args, kwargs: ctrl.gripper_movement(*args, **kwargs),
        "home": lambda args, kwargs: ctrl.joint_movement([0.0, 0.261, -2.27, 0.0, 0.96, 1.5708]),
    }


def start_executor(node):
    exec_ = MultiThreadedExecutor()
    exec_.add_node(node)

    t = threading.Thread(target=exec_.spin, daemon=True)
    t.start()
    return exec_, t


def main(bind_addr="tcp://0.0.0.0:5555"):
    rclpy.init()

    ctrl = KinovaGen3Controller("gen3_remote_server")

    exec_, spin_thread = start_executor(ctrl)

    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.REP)
    sock.bind(bind_addr)
    print(f"[server] listening on {bind_addr}")

    shutting_down = {"flag": False}

    def _shutdown(*_):
        if shutting_down["flag"]:
            return
        shutting_down["flag"] = True
        try:
            sock.close(0)
        except Exception:
            pass
        print("\n[server] shutting down...")

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    dispatch = build_dispatch(ctrl)

    while not shutting_down["flag"]:
        try:
            msg = sock.recv_json(flags=0)
        except zmq.error.ZMQError:
            break

        try:
            cmd = msg.get("cmd")
            args = msg.get("args", [])
            kwargs = msg.get("kwargs", {})
            if cmd == "_ping":
                resp = {"ok": True, "result": "pong", "t": time.time()}
            else:
                fn = dispatch.get(cmd)
                if not fn:
                    raise ValueError(f"Unknown cmd: {cmd}")
                result = fn(args, kwargs)
                resp = resp = {
                    "ok": True,
                    "result": {
                    "reached": bool(getattr(msg, "reached_goal", False)),
                    "stalled": bool(getattr(msg, "stalled", False)),
                    "position": float(getattr(msg, "position", 0.0)) if getattr(msg, "position", None) is not None else None,
                    "effort": float(getattr(msg, "effort", 0.0)) if getattr(msg, "effort", None) is not None else None,
                },
                }
        except Exception as e:
            traceback.print_exc()
            resp = {"ok": False, "error": str(e)}

        try:
            sock.send_json(resp)
        except zmq.error.ZMQError:
            break

    try:
        exec_.shutdown()
    except Exception:
        pass
    try:
        ctrl.destroy_node()
    except Exception:
        pass
    rclpy.shutdown()
    print("[server] bye")


if __name__ == "__main__":
    bind = sys.argv[1] if len(sys.argv) > 1 else "tcp://0.0.0.0:5555"
    main(bind)
