import zmq, json

class KinovaGen3Controller:
    """
    Proxy with the SAME method names as the real controller on A.
    Calls are sent over ZeroMQ to the server on laptop A.
    """
    def __init__(self, server_addr="tcp://10.0.0.2:5555", node_name="gen3_remote_client"):
        self.ctx = zmq.Context.instance()
        self.sock = self.ctx.socket(zmq.REQ)
        self.sock.connect(server_addr)

    def _call(self, cmd, *args, **kwargs):
        self.sock.send_json({"cmd": cmd, "args": list(args), "kwargs": kwargs})
        resp = self.sock.recv_json()
        if not resp.get("ok", False):
            raise RuntimeError(resp.get("error", "Unknown RPC error"))
        return resp.get("result")

    # ---- mirror the real API ----
    def joint_movement(self, *args, **kwargs):
        return self._call("joint_movement", *args, **kwargs)

    def cartesian_movement(self, *args, **kwargs):
        return self._call("cartesian_movement", *args, **kwargs)

    def publish_cartesian_velocity(self, *args, **kwargs):
        return self._call("publish_cartesian_velocity", *args, **kwargs)

    def gripper_movement(self, *args, **kwargs):
        return self._call("gripper_movement", *args, **kwargs)

    # optional helpers
    def home(self):
        return self._call("home")

    def destroy_node(self):
        # no-op for proxy; keep method so your shutdown() code doesn't break
        pass
