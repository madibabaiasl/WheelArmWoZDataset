# openteach/components/recorders/audio.py

import os
import time
import socket
import queue
import signal
from pathlib import Path

import numpy as np
import subprocess
import fcntl

try:
    import soundfile as sf
    import sounddevice as sd
except Exception as e:
    raise RuntimeError(
        "This recorder needs 'soundfile' and 'sounddevice'. Install with:\n"
        "  pip install soundfile sounddevice"
    ) from e

try:
    from opuslib import Decoder, OpusError
except Exception as e:
    raise RuntimeError(
        "HeadsetOpusRecorder needs 'opuslib'. Install with:\n"
        "  pip install opuslib"
    ) from e

from .recorder import Recorder  # same base used by your other recorders
from openteach.utils.files import store_pickle_data


# --------- Shared defaults (match your UDP receiver script) ----------
_OPUS_SR = 48000
_OPUS_CH = 1
_FRAME_MS = 20
_SAMPLES_PER_FRAME = _OPUS_SR * _FRAME_MS // 1000
# --------------------------------------------------------------------


class HeadsetOpusRecorder(Recorder):
    """
    Record an incoming Opus-over-UDP stream (from the headset) to a WAV file.

    - Binds UDP on 0.0.0.0:<port>
    - Decodes each packet with opuslib.Decoder
    - Writes mono PCM16 @ 48 kHz to <storage_path>/<filename>.wav
    - Optional live monitor:
        * PortAudio OutputStream (when available), OR
        * paplay/aplay pipe (Pulse/ALSA) when force_pipe=True or output_device="pipe"
    """

    def __init__(
        self,
        port: int,
        storage_path: str,
        filename: str = "headset_audio",
        header_skip: int = 0,         # set to 8 if your sender prepends an 8-byte header
        play: bool = False,                   # enable live monitor
        output_device=None,                   # sd.OutputStream device index/name, or "pipe"
        prebuffer_frames: int = 3,            # ~60 ms at 20 ms frames
        out_channels: int = 2,
        robot_mod_freq: float | None = None,  # e.g., 90.0 Hz; None/0 disables
        robot_mix: float = 0.8,               # 0..1 wet mix
        crush_bits: int = 0,                  # e.g., 8; 0 disables
        sink_name: str | None = None,         # Pulse sink for paplay (e.g., your laptop speakers)
        force_pipe: bool = False,             # True → skip PortAudio and go straight to paplay/aplay
    ):
        self.port = int(port)
        self.storage_path = storage_path
        self.filename = filename
        self.header_skip = int(header_skip)

        self.notify_component_start(self.filename)

        self._sock = None
        self._dec = Decoder(_OPUS_SR, _OPUS_CH)
        self._wav = None

        # Pipe-based playback (paplay/aplay)
        self._pipe_play = None
        self.sink_name = sink_name
        self.force_pipe = bool(force_pipe)

        self._recorder_file_name = os.path.join(self.storage_path, f"{self.filename}.wav")
        self._metadata_filename = os.path.join(self.storage_path, f"{self.filename}.metadata")

        self.packet_timestamps = []
        self.num_packets = 0
        self.record_start_time = None
        self.record_end_time = None

        self.play = bool(play)
        self.output_device = output_device
        self.prebuffer_frames = int(max(0, prebuffer_frames))
        self.out_channels = int(out_channels)
        self._out_q = queue.Queue(maxsize=200) if self.play else None
        self._out_stream = None
        self._robot = None
        if (robot_mod_freq and robot_mod_freq > 0) or (crush_bits and crush_bits > 0):
            class Robotizer:
                def __init__(self, sr, mod_freq=90.0, mix=0.8, crush_bits=0):
                    self.sr = sr; self.mix = float(np.clip(mix, 0.0, 1.0))
                    self.crush_bits = int(crush_bits) if crush_bits and crush_bits > 0 else 0
                    self.phase = 0.0
                    self.set_freq(mod_freq or 0.0)
                def set_freq(self, f):
                    self.mod_freq = float(max(0.0, f))
                    self.mod_inc = 2.0*np.pi*self.mod_freq/self.sr if self.mod_freq > 0 else 0.0
                def process(self, x_i16):
                    if (self.mod_freq <= 0) and (self.crush_bits <= 0) and (self.mix <= 0):
                        return x_i16
                    x = x_i16.astype(np.float32) / 32768.0; y = x
                    if self.mod_freq > 0:
                        n = x.shape[0]; phase = self.phase + self.mod_inc*np.arange(n, dtype=np.float32)
                        y = x * np.sin(phase, dtype=np.float32); self.phase = (phase[-1] + self.mod_inc) % (2*np.pi)
                    if self.crush_bits > 0:
                        levels = float(2 ** self.crush_bits)
                        y = np.round(y * (levels/2 - 1)) / (levels/2 - 1)
                    out = (1.0 - self.mix) * x + self.mix * y
                    return (np.clip(out, -1.0, 1.0) * 32767.0).astype(np.int16)
            self._robot = Robotizer(_OPUS_SR, robot_mod_freq or 0.0, mix=robot_mix, crush_bits=crush_bits)

    # ---- PortAudio callback (used only if not forcing pipe) ----
    def _out_callback(self, outdata, frames, time_info, status):
        if status:
            import sys; sys.stderr.write(str(status) + "\n")
        outdata.fill(0)
        filled = 0
        while filled < frames:
            try:
                mono_i16 = self._out_q.get_nowait()
            except queue.Empty:
                break
            n = min(len(mono_i16), frames - filled)
            # float32 is broadly compatible
            y = mono_i16[:n].astype(np.float32) / 32768.0
            if outdata.shape[1] == 1:
                outdata[filled:filled+n, 0] = y
            else:  # duplicate mono→stereo
                outdata[filled:filled+n, 0] = y
                outdata[filled:filled+n, 1] = y
            filled += n

    # ---- Pipe-based player helpers (Pulse/ALSA) ----
    def _open_pipe_player(self):
        """
        Start a system audio player that reads raw PCM from stdin.
        Prefer paplay (Pulse), else fall back to aplay (ALSA).
        """
        paplay_cmd = ["paplay", "--raw", "--rate=48000", "--channels=1", "--format=s16le"]
        if self.sink_name:
            paplay_cmd.insert(1, f"--device={self.sink_name}")

        candidates = [
            paplay_cmd,                                   # PulseAudio/PipeWire
            ["aplay", "-f", "S16_LE", "-c", "1", "-r", "48000"],  # ALSA fallback
        ]
        for cmd in candidates:
            try:
                p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
                fd = p.stdin.fileno()
                flags = fcntl.fcntl(fd, fcntl.F_GETFL)
                fcntl.fcntl(fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)
                print(f"[HeadsetOpusRecorder] pipe player started: {' '.join(cmd)} (pid {p.pid})")
                return p
            except Exception as e:
                print(f"[HeadsetOpusRecorder] pipe player failed: {' '.join(cmd)} -> {e}")
        return None

    def _close_pipe_player(self):
        if self._pipe_play:
            try: self._pipe_play.stdin.close()
            except Exception: pass
            try: self._pipe_play.terminate()
            except Exception: pass
            self._pipe_play = None

    def _open_output(self):
        if not self.play:
            return

        # >>> FORCE PIPE PLAYBACK (skip PortAudio entirely) <<<
        if self.force_pipe or str(self.output_device).lower() in ("pipe", "paplay", "aplay"):
            print("[HeadsetOpusRecorder] Forcing paplay/aplay (pipe) playback…")
            self._pipe_play = self._open_pipe_player()
            if not self._pipe_play:
                raise RuntimeError("No available audio output (paplay/aplay failed).")
            return
        # >>> end force pipe block <<<

        # ---- PortAudio path (best-effort) ----
        import sounddevice as sd
        from sounddevice import PortAudioError

        # normalize requested device to a scalar
        req = self.output_device
        if isinstance(req, (list, tuple)):
            req = (req[1] if len(req) > 1 and req[1] is not None else req[0] if len(req) else None)

        # try mixer/virtuals first, then HDMI, then indices/default
        dev_candidates = [req, "pulse", "default", "hdmi", 10, 13, 1, 2, 3, 4, None]
        last_err = None

        def _devinfo(dev):
            try:
                return sd.query_devices(dev, kind="output")
            except Exception:
                return None

        for dev in dev_candidates:
            info = _devinfo(dev)
            if info is None:
                continue
            max_out = int(info.get("max_output_channels", 0))
            if max_out < 1:
                continue

            # clamp ch to device capability
            req_ch = int(getattr(self, "out_channels", 2) or 2)
            ch = max(1, min(req_ch, max_out))

            # try SRs: 48k (Opus), 44.1k (Pulse default), device default
            sr_cands = []
            for sr in (48000, 44100, info.get("default_samplerate")):
                try:
                    s = int(sr)
                    if s > 0 and s not in sr_cands:
                        sr_cands.append(s)
                except Exception:
                    pass

            print(f"[HeadsetOpusRecorder] Trying output → {dev} | {info['name']} (max_out={max_out}) ch={ch} sr={sr_cands}")

            for sr in sr_cands:
                for dtype in ("float32", "int16"):
                    try:
                        self._out_stream = sd.OutputStream(
                            device=dev, samplerate=sr,
                            channels=ch, dtype=dtype,
                            blocksize=0, callback=self._out_callback,
                        )
                        self._out_stream.start()
                        self.out_channels = ch
                        print(f"[HeadsetOpusRecorder] Output OK on {info['name']} dev={dev} @ {sr} Hz, ch={ch}, dtype={dtype}")
                        # brief prebuffer
                        print(f"[HeadsetOpusRecorder] prebuffering {self.prebuffer_frames} frames…")
                        t0 = time.time()
                        while self._out_q.qsize() < self.prebuffer_frames and (time.time() - t0) < 2.0:
                            sd.sleep(5)
                        print(f"[HeadsetOpusRecorder] playing (ch={self.out_channels})")
                        return
                    except Exception as e:
                        print(f"[HeadsetOpusRecorder] Output open failed dev={dev} sr={sr} dtype={dtype}: {e}")
                        last_err = e
                        self._out_stream = None

        # ---- last resort: pipe to paplay/aplay ----
        print("[HeadsetOpusRecorder] Falling back to paplay/aplay…")
        self._pipe_play = self._open_pipe_player()
        if self._pipe_play:
            return
        raise RuntimeError(f"Failed to open any output stream: {last_err}")

    def _open_udp(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 20)
        sock.bind(("0.0.0.0", self.port))
        sock.settimeout(0.5)
        return sock

    def _open_file(self):
        os.makedirs(self.storage_path, exist_ok=True)
        return sf.SoundFile(
            self._recorder_file_name, mode="w",
            samplerate=_OPUS_SR, channels=_OPUS_CH, subtype="PCM_16"
        )

    def stream(self):
        self._sock = self._open_udp()
        self._wav = self._open_file()
        print(f"[HeadsetOpusRecorder] UDP :{self.port} → {self._recorder_file_name}")

        if self.play:
            try:
                self._open_output()
            except Exception as e:
                print(f"[HeadsetOpusRecorder] Playback disabled: {e}")
                self.play = False

        self.record_start_time = time.time()
        try:
            while True:
                try:
                    data, addr = self._sock.recvfrom(4096)
                except socket.timeout:
                    continue
                if not data or len(data) <= self.header_skip:
                    continue
                payload = data[self.header_skip:]

                try:
                    pcm = self._dec.decode(payload, frame_size=_SAMPLES_PER_FRAME, decode_fec=False)
                except OpusError as e:
                    print(f"[HeadsetOpusRecorder] Opus decode error: {e} (payload={len(payload)}B)")
                    continue

                # Live monitor (PortAudio or pipe)
                if self.play:
                    pcm_np = np.frombuffer(pcm, dtype=np.int16)
                    if self._robot is not None:
                        pcm_np = self._robot.process(pcm_np)

                    if self._out_stream is not None:
                        try:
                            if self._out_q.qsize() >= self._out_q.maxsize - 2:
                                _ = self._out_q.get_nowait()
                            self._out_q.put_nowait(pcm_np)
                        except queue.Full:
                            pass
                    elif self._pipe_play is not None:
                        try:
                            self._pipe_play.stdin.write(pcm_np.tobytes())
                        except (BrokenPipeError, OSError):
                            print("[HeadsetOpusRecorder] paplay/aplay died; disabling fallback")
                            self._close_pipe_player()
                            self.play = False

                # Write to WAV
                mono = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
                self._wav.write(mono)

                # Stats
                self.packet_timestamps.append(time.time())
                self.num_packets += 1

        except KeyboardInterrupt:
            pass
        finally:
            self.record_end_time = time.time()
            try:
                if self._sock:
                    self._sock.close()
            finally:
                if self._wav:
                    self._wav.flush()
                    self._wav.close()
                    print(f"[HeadsetOpusRecorder] Saved {Path(self._recorder_file_name).resolve()}")

            # Stop PortAudio, close pipe, and write metadata
            try:
                if self._out_stream:
                    self._out_stream.stop()
                    self._out_stream.close()
                self._close_pipe_player()

                self._add_metadata(self.num_packets)
                duration = (self.record_end_time - self.record_start_time) if (self.record_end_time and self.record_start_time) else None
                self.metadata.update({
                    "timestamps": self.packet_timestamps,      # arrival times per opus frame
                    "audio_samplerate_hz": _OPUS_SR,
                    "audio_channels": _OPUS_CH,
                    "audio_subtype": "PCM_16",
                    "num_packets": self.num_packets,
                    "duration_sec": duration,
                    "recorder_udp_port": self.port,
                    "header_skip_bytes": self.header_skip,
                    "decoder_frame_ms": _FRAME_MS,
                })
                store_pickle_data(self._metadata_filename, self.metadata)
                print(f"Stored the metadata in {self._metadata_filename}")
            except Exception as e:
                print(f"[HeadsetOpusRecorder] Metadata write failed: {e}")


class LaptopMicWavRecorder(Recorder):
    """
    Record the laptop microphone to WAV, with optional file segmentation and graceful drain.
    """

    def __init__(
        self,
        storage_path: str,
        filename: str = "laptop_mic",
        samplerate: int = 48000,
        channels: int = 1,
        device=None,                  # index or name, or None for default input
        subtype: str = "PCM_16",
        segment_seconds: float | None = None,  # rotate to a new file every N seconds
        drain_seconds: float = 2.0,            # on shutdown, flush up to this many seconds
    ):
        self.storage_path = storage_path
        self.filename = filename
        self.samplerate = int(samplerate)
        self.channels = int(channels)
        self.device = device
        self.subtype = subtype
        self.segment_seconds = segment_seconds
        self.drain_seconds = float(drain_seconds)

        self._q: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=64)
        self._stream = None
        self._wav = None
        self._cur_path: Path | None = None
        self._segment_start = None

        self._base_stem = filename
        self.notify_component_start(self.filename)
        
        self._recorder_file_name = None
        self._metadata_filename = os.path.join(self.storage_path, f"{self.filename}.metadata")
        self.audio_timestamps = []
        self.num_chunks = 0
        self.sample_count = 0
        self.record_start_time = None
        self.record_end_time = None
        self.device_info = None

        self._dtype = "float32"
        self.blocksize = 0    # let backend choose; most compatible

    # ---------- helpers ----------
    @staticmethod
    def _timestamped_name(prefix: str, suffix: str = ".wav") -> str:
        from datetime import datetime
        return f"{prefix}_{datetime.now().strftime('%Y%m%d-%H%M%S')}{suffix}"

    def _make_out_path(self) -> Path:
        os.makedirs(self.storage_path, exist_ok=True)
        if self.segment_seconds:
            name = self._timestamped_name(self._base_stem)
        else:
            name = f"{self._base_stem}.wav"
        return Path(self.storage_path) / name

    def _callback(self, indata, frames, time_info, status):
        if status:
            print(status, flush=True)
        try:
            self._q.put_nowait(indata.copy())
        except queue.Full:
            pass

    # def _open_stream(self):
    #     import sounddevice as sd
    #     from sounddevice import PortAudioError

    #     # ---- normalize to scalar device (None/int/str) ----
    #     # dev0 = getattr(self, "device", None)
    #     # if isinstance(dev0, (list, tuple)):
    #     #     dev0 = dev0[0] if len(dev0) > 0 else None

    #     # # Candidate input devices (ordered)
    #     # dev_candidates = [dev0, "pulse", "default", 0, 5, None]
    #     import os

    #     # --- If user passed a Pulse source name, set it here (works per-process) ---
    #     # e.g. "alsa_input.usb-Jieli_Technology_USB_Composite_Device_433130353331342E-00.mono-fallback"
    #     pulse_source = None
    #     if isinstance(self.device, str) and self.device.startswith("alsa_input."):
    #         pulse_source = self.device
    #         os.environ["PULSE_SOURCE"] = pulse_source       # IMPORTANT: set before importing sounddevice
    #         pa_device = "pulse"                             # open PortAudio on Pulse host
    #     else:
    #         pa_device = self.device                         # could be 10 (pulse index), "pulse", or None

    #     import sounddevice as sd
    #     from sounddevice import PortAudioError

    #     # dev0 = getattr(self, "device", None)
    #     print(f"[LaptopRecorder]: device is {pa_device}")
    #     dev0 = pa_device
    #     if isinstance(dev0, (list, tuple)):
    #         dev0 = dev0[0] if len(dev0) > 0 else None
    #     # Only try what you asked for -> pulse -> default. No numeric ALSA fallbacks.
    #     dev_candidates = [dev0, "pulse", "default"]

    #     # Reasonable defaults if missing
    #     if not hasattr(self, "_dtype"):
    #         self._dtype = "float32"
    #     if not hasattr(self, "blocksize"):
    #         self.blocksize = 0  # let backend decide

    #     last_err = None

    #     # Try devices
    #     for dev in dev_candidates:
    #         try:
    #             info = sd.query_devices(dev, kind="input")
    #         except Exception:
    #             continue

    #         # Cap channels to device capability
    #         ch = min(self.channels, int(info.get("max_input_channels", self.channels)))
    #         if ch < 1:
    #             continue

    #         # Build samplerate candidates
    #         default_sr = int(info.get("default_samplerate") or 48000)
    #         sr_pref = []
    #         if getattr(self, "samplerate", None):
    #             sr_pref.append(int(self.samplerate))
    #         sr_pref += [default_sr, 48000, 44100, 32000, 22050, 16000, 8000]
    #         seen = set()
    #         sr_cands = [sr for sr in sr_pref if sr and sr > 0 and (sr not in seen and not seen.add(sr))]

    #         # Try dtype/rate combos
    #         for dtype in ("int16", "float32"):
    #             for sr in sr_cands:
    #                 try:
    #                     st = sd.InputStream(
    #                         device=dev,
    #                         channels=ch,
    #                         samplerate=sr,
    #                         dtype=dtype,
    #                         blocksize=self.blocksize,
    #                         callback=self._callback,
    #                     )
    #                     st.start()
    #                     # success
    #                     self._stream = st
    #                     self.device = dev
    #                     self.device_info = dict(info)
    #                     self.channels = ch
    #                     self.samplerate = sr
    #                     self._dtype = dtype
    #                     print(f"[LaptopMicWavRecorder] Using device={dev!r} name='{sd.query_devices(dev or 'pulse')['name']}' "
    #                     f"@ {sr} Hz, ch={ch}, dtype={dtype}")
    #                     return
    #                 except PortAudioError as e:
    #                     last_err = e
    #                     self._stream = None
    #                     continue

    #     raise RuntimeError(f"Could not open input stream at any supported device/rate; last error: {last_err}")

    def _open_stream(self):
        import os

        # 0) If user passed a Pulse SOURCE name, set it here (same process), and open PortAudio on "pulse"
        pulse_source = None
        dev0 = getattr(self, "device", None)
        if isinstance(dev0, str) and dev0.startswith("alsa_input."):
            pulse_source = dev0
            os.environ["PULSE_SOURCE"] = pulse_source
            dev0 = "pulse"  # we will open the Pulse device

        import sounddevice as sd
        from sounddevice import PortAudioError

        # 1) Prefer the PulseAudio host API if present
        try:
            pulse_api_idx = next((i for i, a in enumerate(sd.query_hostapis())
                                if "Pulse" in a.get("name", "")), None)
            if pulse_api_idx is not None:
                sd.default.hostapi = pulse_api_idx
        except Exception:
            pass

        # 2) Build device candidates: requested → "pulse" → "default"
        if isinstance(dev0, (list, tuple)):
            dev0 = dev0[0] if dev0 else None
        dev_candidates = [dev0, "pulse", "default"]

        # defaults
        if not hasattr(self, "_dtype"):
            self._dtype = "float32"
        if not hasattr(self, "blocksize"):
            self.blocksize = 0

        last_err = None

        for dev in dev_candidates:
            if dev is None:
                continue
            try:
                # Align I/O to the same device to avoid ALSA duplex mismatch checks
                try:
                    sd.default.device = (dev, dev)   # input=dev, output=dev (we won't actually output)
                except Exception:
                    pass

                info = sd.query_devices(dev, kind="input")
            except Exception:
                continue

            ch = min(self.channels, int(info.get("max_input_channels", self.channels)))
            if ch < 1:
                continue

            # Prefer device default SR, then common rates
            default_sr = int(info.get("default_samplerate") or 48000)
            sr_pref = []
            if getattr(self, "samplerate", None):
                sr_pref.append(int(self.samplerate))
            sr_pref += [default_sr, 48000, 44100, 32000, 22050, 16000, 8000]
            seen = set()
            sr_cands = [sr for sr in sr_pref if sr and sr > 0 and (sr not in seen and not seen.add(sr))]

            for dtype in ("int16", "float32"):   # Pulse is often happiest with int16
                for sr in sr_cands:
                    try:
                        st = sd.InputStream(
                            device=dev, channels=ch, samplerate=sr, dtype=dtype,
                            blocksize=self.blocksize, callback=self._callback,
                        )
                        st.start()
                        # success
                        self._stream = st
                        self.device = dev
                        self.device_info = dict(info)
                        self.channels = ch
                        self.samplerate = sr
                        self._dtype = dtype
                        try:
                            hostapi_name = sd.query_hostapis()[sd.default.hostapi]["name"]
                        except Exception:
                            hostapi_name = "unknown"
                        print(f"[LaptopMicWavRecorder] Using device={dev!r} name='{sd.query_devices(dev)['name']}' "
                            f"@ {sr} Hz, ch={ch}, dtype={dtype}, pulse_source={pulse_source!r}, hostapi='{hostapi_name}'")
                        return
                    except PortAudioError as e:
                        last_err = e
                        self._stream = None
                        continue

        # 3) Last resort: fall back to parec if a Pulse source was provided
        if pulse_source:
            print("[LaptopMicWavRecorder] PortAudio open failed; falling back to parec…")
            self._start_parec_reader(pulse_source)
            return

        raise RuntimeError(f"Could not open input stream at any supported device/rate; last error: {last_err}")

    def _start_parec_reader(self, source_name: str):
        import subprocess, threading, numpy as np

        # Format: 16-bit PCM, mono, 48 kHz (matches your USB source)
        rate = int(getattr(self, "samplerate", 48000) or 48000)
        ch = int(getattr(self, "channels", 1) or 1)

        # parec command: read from Pulse source -> raw s16le to stdout
        cmd = [
            "parec",
            "-d", source_name,             # exact Pulse source (e.g., alsa_input.usb-Jieli_...-mono-fallback)
            "--format=s16le",
            "--rate", str(rate),
            "--channels", str(ch),
            "--raw",
            "--latency-msec=10",
        ]

        # Launch
        self._parec_proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=0)
        bytes_per_sample = 2
        frame_bytes = ch * bytes_per_sample
        chunk_frames = 1024

        def _reader():
            try:
                while True:
                    buf = self._parec_proc.stdout.read(chunk_frames * frame_bytes)
                    if not buf:
                        break
                    arr = np.frombuffer(buf, dtype=np.int16)
                    if ch > 1:
                        arr = arr.reshape(-1, ch)
                    self._q.put(arr)
            except Exception:
                pass

        self._reader_thread = threading.Thread(target=_reader, daemon=True)
        self._reader_thread.start()

        # Mark “stream” as active and set dtype/subtype consistently
        self._stream = True          # sentinel so the rest of your code runs
        self._dtype = "int16"
        if not hasattr(self, "subtype"):
            self.subtype = "PCM_16"
        self.channels = ch
        self.samplerate = rate
        self.device_info = {"name": f"parec:{source_name}", "hostapi": "Pulse (parec)", "max_input_channels": ch, "default_samplerate": rate}
        print(f"[LaptopMicWavRecorder] Using parec source='{source_name}' @ {rate} Hz, ch={ch}, dtype=int16")

    def _stop_parec(self):
        try:
            if getattr(self, "_parec_proc", None):
                self._parec_proc.terminate()
                try:
                    self._parec_proc.wait(timeout=1)
                except Exception:
                    self._parec_proc.kill()
        except Exception:
            pass

    def _open_new_file(self):
        self._cur_path = self._make_out_path()
        self._recorder_file_name = str(self._cur_path)
        self._wav = sf.SoundFile(
            self._cur_path, mode="w",
            samplerate=self.samplerate, channels=self.channels, subtype=self.subtype
        )
        self._segment_start = time.time()
        print(f"[LaptopMicWavRecorder] Recording → {self._cur_path} ({self.channels} ch @ {self.samplerate} Hz, {self.subtype})")

    def _maybe_rotate(self):
        if not self.segment_seconds:
            return
        if (time.time() - self._segment_start) >= self.segment_seconds:
            self._wav.flush()
            self._wav.close()
            print(f"[LaptopMicWavRecorder] Segment saved: {self._cur_path.resolve()}")
            self._open_new_file()

    # ---------- main loop ----------
    def stream(self):
        self._open_stream()
        self._open_new_file()
        self.record_start_time = time.time()
        try:
            while True:
                try:
                    chunk = self._q.get(timeout=0.25)
                    self._wav.write(chunk)
                    # ---- stats ----
                    self.num_chunks += 1
                    self.sample_count += len(chunk)
                    self.audio_timestamps.append(time.time())
                except queue.Empty:
                    pass
                self._maybe_rotate()
        except KeyboardInterrupt:
            pass
        finally:
            try:
                if self._stream:
                    self._stream.stop()
                    self._stop_parec()
            except Exception:
                pass

            deadline = time.time() + self.drain_seconds
            while time.time() < deadline:
                try:
                    chunk = self._q.get_nowait()
                    self._wav.write(chunk)
                    self.num_chunks += 1
                    self.sample_count += len(chunk)
                    self.audio_timestamps.append(time.time())
                except queue.Empty:
                    break
                except Exception:
                    break

            try:
                if self._stream:
                    self._stream.close()
            except Exception:
                pass
            try:
                if self._wav:
                    self._wav.flush()
                    self._wav.close()
                    print(f"[LaptopMicWavRecorder] Saved: {self._cur_path.resolve()}")
            except Exception:
                pass

            self.record_end_time = time.time()

            # ---- metadata (mirror image.py) ----
            try:
                self._add_metadata(self.num_chunks)
                duration = (self.record_end_time - self.record_start_time) if (self.record_end_time and self.record_start_time) else None
                dev_info = self.device_info or {}
                self.metadata.update({
                    "timestamps": self.audio_timestamps,
                    "audio_samplerate_hz": self.samplerate,
                    "audio_channels": self.channels,
                    "audio_subtype": self.subtype,
                    "num_chunks": self.num_chunks,
                    "num_samples_written": int(self.sample_count),
                    "duration_sec": duration,
                    "device": {
                        "index": self.device if isinstance(self.device, int) else None,
                        "name": dev_info.get("name"),
                        "hostapi": dev_info.get("hostapi"),
                        "default_samplerate": dev_info.get("default_samplerate"),
                        "max_input_channels": dev_info.get("max_input_channels"),
                    },
                    "segment_seconds": self.segment_seconds,
                    "drain_seconds": self.drain_seconds,
                })
                store_pickle_data(self._metadata_filename, self.metadata)
                print(f"Stored the metadata in {self._metadata_filename}")
            except Exception as e:
                print(f"[LaptopMicWavRecorder] Metadata write failed: {e}")
