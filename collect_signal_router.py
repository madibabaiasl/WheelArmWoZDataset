import zmq
import time
import signal
import sys
import socket
import threading

from openteach.constants import *

# ----------------------------
# Config
# ----------------------------
A_BIND_IP = "0.0.0.0"     # listen on laptop A
B_IP = "10.0.0.2"

# If True, print every forwarded message
VERBOSE = True


def log(msg: str):
    print(f"[relay] {msg}", flush=True)

class UdpAudioForwarder:
    def __init__(self, bind_ip: str, listen_port: int, dest_ip: str, dest_port: int, bufsize: int = 65535):
        self.bind_ip = bind_ip
        self.listen_port = int(listen_port)
        self.dest_ip = dest_ip
        self.dest_port = int(dest_port)
        self.bufsize = int(bufsize)
        self._running = threading.Event()
        self._running.set()
        self._recv_sock = None
        self._send_sock = None
        self._thread = None
        self.packet_count = 0
        self.byte_count = 0

    def start(self):
        self._recv_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._recv_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._recv_sock.bind((self.bind_ip, self.listen_port))
        self._recv_sock.settimeout(0.5)

        self._send_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        log(
            f"forwarding AUDIO UDP from udp://{self.bind_ip}:{self.listen_port} "
            f"to udp://{self.dest_ip}:{self.dest_port}"
        )

    def _loop(self):
        while self._running.is_set():
            try:
                data, addr = self._recv_sock.recvfrom(self.bufsize)
            except socket.timeout:
                continue
            except OSError as e:
                if self._running.is_set():
                    log(f"audio relay recv error: {e}")
                break

            try:
                self._send_sock.sendto(data, (self.dest_ip, self.dest_port))
                self.packet_count += 1
                self.byte_count += len(data)
                if VERBOSE and (self.packet_count <= 5 or self.packet_count % 200 == 0):
                    log(
                        f"forwarded AUDIO packet #{self.packet_count} "
                        f"({len(data)} bytes) from {addr[0]}:{addr[1]}"
                    )
            except OSError as e:
                log(f"audio relay send error: {e}")
                time.sleep(0.05)

    def stop(self):
        self._running.clear()
        for s in (self._recv_sock, self._send_sock):
            try:
                if s is not None:
                    s.close()
            except Exception:
                pass
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        if self.packet_count:
            log(f"audio relay stopped after {self.packet_count} packets / {self.byte_count} bytes")

def main():
    ctx = zmq.Context.instance()

    # --- Receive from VR on laptop A ---
    recv_collect = ctx.socket(zmq.PULL)
    recv_collect.setsockopt(zmq.RCVHWM, 100)
    recv_collect.bind(f"tcp://{A_BIND_IP}:{COLLECT_PORT}")

    recv_session = ctx.socket(zmq.PULL)
    recv_session.setsockopt(zmq.RCVHWM, 100)
    recv_session.bind(f"tcp://{A_BIND_IP}:{SESSION_PORT}")

    # --- Forward to laptop B ---
    send_collect = ctx.socket(zmq.PUSH)
    send_collect.setsockopt(zmq.SNDHWM, 100)
    send_collect.connect(f"tcp://{DEVICE_B_IP}:{COLLECT_PORT}")

    send_session = ctx.socket(zmq.PUSH)
    send_session.setsockopt(zmq.SNDHWM, 100)
    send_session.connect(f"tcp://{DEVICE_B_IP}:{SESSION_PORT}")

    poller = zmq.Poller()
    poller.register(recv_collect, zmq.POLLIN)
    poller.register(recv_session, zmq.POLLIN)

    running = True
    audio_forwarder = None

    if AUDIO_PORT is not None:
        audio_forwarder = UdpAudioForwarder(
            bind_ip=A_BIND_IP,
            listen_port=AUDIO_PORT,
            dest_ip=DEVICE_B_IP,
            dest_port=AUDIO_PORT,
            bufsize=UDP_BUFFER_SIZE,
        )
        audio_forwarder.start()
    else:
        log("AUDIO relay disabled (set AUDIO_PORT to enable UDP headset audio forwarding)")


    def handle_exit(signum, frame):
        nonlocal running
        log("shutting down")
        running = False

    signal.signal(signal.SIGINT, handle_exit)
    signal.signal(signal.SIGTERM, handle_exit)

    log(f"listening for VR collect on tcp://{A_BIND_IP}:{COLLECT_PORT}")
    log(f"listening for VR session on tcp://{A_BIND_IP}:{SESSION_PORT}")
    log(f"forwarding collect to tcp://{B_IP}:{COLLECT_PORT}")
    log(f"forwarding session to tcp://{B_IP}:{SESSION_PORT}")

    try:
        while running:
            events = dict(poller.poll(timeout=500))

            if recv_session in events:
                try:
                    msg = recv_session.recv_string(flags=zmq.NOBLOCK).strip()
                    send_session.send_string(msg)
                    if VERBOSE:
                        log(f"forwarded SESSION -> {msg!r}")
                except zmq.Again:
                    pass

            if recv_collect in events:
                try:
                    msg = recv_collect.recv_string(flags=zmq.NOBLOCK).strip()
                    send_collect.send_string(msg)
                    if VERBOSE:
                        log(f"forwarded COLLECT -> {msg!r}")
                except zmq.Again:
                    pass

    finally:
        if audio_forwarder is not None:
            audio_forwarder.stop()
        recv_collect.close(0)
        recv_session.close(0)
        send_collect.close(0)
        send_session.close(0)
        ctx.term()
        log("closed cleanly")


if __name__ == "__main__":
    main()