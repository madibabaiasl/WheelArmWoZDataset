import zmq
import hydra
from omegaconf import OmegaConf
from openteach.components import Collector

# IP&ports configuration
LAPTOP_BIND_IP = "0.0.0.0"   
COLLECT_PORT   = 8126        
SESSION_PORT   = 8123       
host = LAPTOP_BIND_IP

def _recv_str(sock):
    try:
        return sock.recv(flags=zmq.NOBLOCK).decode().strip()
    except zmq.Again:
        return None

@hydra.main(version_base='1.2', config_path='configs', config_name='collect_data')
def main(configs):
    print("[collector-gui] configs:")
    print(OmegaConf.to_yaml(configs))

    # ZMQ setup
    ctx = zmq.Context.instance()
    collect = ctx.socket(zmq.PULL); collect.setsockopt(zmq.RCVHWM, 100); collect.bind(f"tcp://{host}:{COLLECT_PORT}")
    session = ctx.socket(zmq.PULL); session.setsockopt(zmq.RCVHWM, 100); session.bind(f"tcp://{host}:{SESSION_PORT}")

    poller = zmq.Poller()
    poller.register(collect, zmq.POLLIN)
    poller.register(session, zmq.POLLIN)

    current_session = None
    running = False
    collector = None

    print(f"[collector-gui] listening collect @ tcp://{host}:{COLLECT_PORT}")
    print(f"[collector-gui] listening session @ tcp://{host}:{SESSION_PORT}")
    print("[collector-gui] workflow: Send session text → Start (1) → Stop (0)")

    try:
        while True:
            events = dict(poller.poll(timeout=500))  # 0.5s

            # Update session name if sent
            if session in events:
                s = _recv_str(session)
                if s:
                    current_session = s
                    print(f"[collector-gui] session='{current_session}'")

            # Start/Stop recording
            if collect in events:
                v = _recv_str(collect)
                if v is None:
                    continue

                if v == "1":
                    if running:
                        print("[collector-gui] already running; ignoring start")
                        continue
                    # Start with provided session folder name if available
                    sess_name = current_session.strip() if current_session else None
                    print(f"[collector-gui] START recording; session={sess_name or '(default)'}")
                    # NOTE: Collector must accept session_name and have stop()
                    collector = Collector(configs, configs.demo_num, session_name=sess_name)
                    for p in collector.get_processes():
                        p.start()
                    running = True

                elif v == "0":
                    if not running:
                        print("[collector-gui] not running; ignoring stop")
                        continue
                    print("[collector-gui] STOP recording (graceful)")
                    collector.stop()     # sends SIGINT to child recorders
                    running = False
                    collector = None

                else:
                    print(f"[collector-gui] unknown collect value: {v}")

    except KeyboardInterrupt:
        print("\n[collector-gui] shutting down…")
        if running and collector:
            collector.stop()

if __name__ == '__main__':
    main()
