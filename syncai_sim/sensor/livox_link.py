"""The sim as a Livox MID360 / MID360s, on the lidar's own UDP protocol.

With `lidar_link: livox` the sim's lidar is a device on the network: the
workspace's real livox_ros_driver2 (Livox-SDK2 underneath) finds it, configures
it and reads its point and IMU packets exactly as it reads the robot's, and
everything after the driver — Point-LIO, the localizer, lio_bridge, so the
robot's pose in the map and every planner that starts from it — runs on the
sim. Nothing in the workspace changes; the sim stays UDP only (no ROS).

The protocol, as the vendored Livox-SDK2 and livox_ros_driver2 speak it
(paths under the workspace's src/third-party/):

* Control frames (`encode_frame`): a 24-byte header — 0xAA, version 0, length,
  seq, cmd_id, cmd_type (0 request / 1 ACK), sender_type, 6 reserved, CRC-16
  /CCITT-FALSE of bytes 0..17, CRC-32 of the payload (0 for none)
  (sdk_core/comm/sdk_protocol.{h,cpp}). A frame with a bad CRC is dropped.
  An ACK echoes the request's seq and cmd_id: the SDK matches by seq.
* Discovery: the SDK broadcasts 0x0000 to 255.255.255.255:56000 from
  host_ip:56000 every second; the device answers each one, from its port
  56000, with ret 0, its dev_type (MID360 9, MID360s 35 — the bringup's JSON
  is keyed by type, so a wrong one is ignored silently), a serial, its IP
  and its command port 56100 (comm/define.h DetectionData). Keep answering:
  the first answer registers it, the second asks the firmware type, a later
  one sends the configuration (general_command_handler.cpp).
* Commands on 56100, each answered: 0x0101 asking key 0x8010 -> firmware type
  1 (app; 0 = loader, and the configuration is never sent); 0x0100 (a
  key-value list) -> ret 0, error key 0. The SDK's configuration names the
  host's push / point / IMU addresses (keys 5, 6, 7); the driver then sets
  the data type, the scan pattern, the attitude, the work mode and IMU
  enable, and publishes only once those are acknowledged.
* Data: points from 56300 to the host's 56301, IMU from 56400 to 56401, each
  packet a 36-byte header (`DATA_HEADER`) and its payload, the SDK checking
  nothing (data_handler.cpp). Points are Cartesian high precision (int32 mm
  x y z, reflectivity, tag), 96 a packet as the real lidar sends; the driver
  times point i at timestamp + i * time_interval / dot_num, so a packet's
  points have to be close to evenly spaced, which the recorded pattern's are.
  IMU: gyro rad/s then acceleration in g (Point-LIO multiplies by 9.81).

Time: `time_type` 1 (PTP), the device's own nanoseconds. With 0 the driver
throws the device time away and stamps each packet as it arrives, and a scan
sent at its end would put 100 ms of firing times into a few milliseconds.
With 1 it cuts a frame when a packet's last point falls on a whole 100 ms
(pub_handler.cpp CheckTimer), so each scan carries a one-point packet on
every such boundary it spans (`MARKER`, a miss the stack drops). The device
clock is the host's epoch clock, tied to sim time (`DeviceClock`): Point-LIO
stamps the pose with it while lio_bridge stamps its TF with the host's
`now()`, and the two have to agree for the TF tree to join.

The robot's points are in the lidar's own frame (lidar_top), each at the
moment it was fired, not motion-compensated (Point-LIO deskews with the IMU);
a miss goes out as (0, 0, 0), as the MID360 sends it, keeping the packets
evenly timed. The IMU is lidar_top_imu's (Simulator.lidar_imu).
"""

from __future__ import annotations

import binascii
import socket
import struct
import time
import zlib
from dataclasses import dataclass

import numpy as np

SOF = 0xAA
CMD_TYPE_REQUEST, CMD_TYPE_ACK = 0, 1
SENDER_HOST, SENDER_LIDAR = 0, 1

CMD_DISCOVERY = 0x0000
CMD_PARAMS = 0x0100  # WorkModeControl: every key-value parameter set
CMD_QUERY = 0x0101  # GetInternalInfo
KEY_FW_TYPE = 0x8010
KEY_PUSH_TARGET, KEY_POINT_TARGET, KEY_IMU_TARGET = 0x0005, 0x0006, 0x0007

PORT_DETECT, PORT_CMD, PORT_POINT, PORT_IMU = 56000, 56100, 56300, 56400
HOST_PORT_POINT, HOST_PORT_IMU = 56301, 56401

DEV_TYPES = {"mid360": 9, "mid360s": 35}  # livox_lidar_def.h

DATA_IMU, DATA_CARTESIAN_HIGH = 0, 1
TIME_PTP = 1

HEADER = struct.Struct("<BBHIHBB6sHI")  # 24 bytes, sdk_protocol.h SdkPacket
DATA_HEADER = struct.Struct("<BHHHHBBB12sIQ")  # 36 bytes, LivoxLidarEthernetPacket
POINT = np.dtype([("x", "<i4"), ("y", "<i4"), ("z", "<i4"), ("reflectivity", "u1"),
                  ("tag", "u1")])  # LivoxLidarCartesianHighRawPoint, 14 bytes
IMU = struct.Struct("<6f")  # gyro xyz rad/s, acc xyz g

FRAME_NS = 100_000_000  # the driver's publish_freq 10 Hz: frames cut on whole 100 ms
G = 9.81


# -- control frames ------------------------------------------------------------
@dataclass(frozen=True)
class Frame:
    seq: int
    cmd_id: int
    cmd_type: int
    payload: bytes


def encode_frame(cmd_id: int, cmd_type: int, seq: int, payload: bytes = b"",
                 sender: int = SENDER_LIDAR) -> bytes:
    crc32 = zlib.crc32(payload) if payload else 0
    head = HEADER.pack(SOF, 0, HEADER.size + len(payload), seq, cmd_id, cmd_type, sender,
                       bytes(6), 0, crc32)
    crc16 = binascii.crc_hqx(head[:18], 0xFFFF)
    return head[:18] + struct.pack("<H", crc16) + head[20:] + payload


def decode_frame(buf: bytes) -> Frame | None:
    """A control frame, or None for anything the SDK itself would drop."""
    if len(buf) < HEADER.size:
        return None
    sof, version, length, seq, cmd_id, cmd_type, _, _, crc16, crc32 = HEADER.unpack_from(buf)
    if sof != SOF or version != 0 or length < HEADER.size or length > len(buf):
        return None
    if binascii.crc_hqx(buf[:18], 0xFFFF) != crc16:
        return None
    payload = bytes(buf[HEADER.size:length])
    if crc32 != (zlib.crc32(payload) if payload else 0):
        return None
    return Frame(seq=seq, cmd_id=cmd_id, cmd_type=cmd_type, payload=payload)


def detection_payload(dev_type: int, serial: str, ip: str) -> bytes:
    """DetectionData: ret, dev_type, sn[16] (NUL-terminated), ip, command port."""
    sn = serial.encode("ascii")[:15].ljust(16, b"\0")
    return struct.pack("<BB16s4sH", 0, dev_type, sn, socket.inet_aton(ip), PORT_CMD)


def query_ack_payload(keys: list[int]) -> bytes:
    """GetInternalInfo's answer: ret 0 and the keys asked that the sim knows.
    Only the firmware type is ever asked (by the SDK, before configuring):
    1 = app firmware."""
    known = {KEY_FW_TYPE: b"\x01"}
    kvs = [(k, known[k]) for k in keys if k in known]
    out = struct.pack("<BH", 0, len(kvs))
    for k, v in kvs:
        out += struct.pack("<HH", k, len(v)) + v
    return out


def parse_query_keys(payload: bytes) -> list[int]:
    """GetInternalInfo's request: key_num, reserved, then the keys alone."""
    if len(payload) < 4:
        return []
    n = struct.unpack_from("<H", payload)[0]
    return [struct.unpack_from("<H", payload, 4 + 2 * i)[0] for i in range(n)
            if 4 + 2 * i + 2 <= len(payload)]


def parse_kv(payload: bytes) -> dict[int, bytes]:
    """A parameter set's key-value list: key_num, reserved, {key, len, value}..."""
    out: dict[int, bytes] = {}
    if len(payload) < 4:
        return out
    n = struct.unpack_from("<H", payload)[0]
    off = 4
    for _ in range(n):
        if off + 4 > len(payload):
            break
        key, ln = struct.unpack_from("<HH", payload, off)
        out[key] = payload[off + 4:off + 4 + ln]
        off += 4 + ln
    return out


def host_target(value: bytes) -> tuple[str, int] | None:
    """HostIpInfoValue (ip[4], host port, lidar port) -> (ip, host port)."""
    if len(value) < 6:
        return None
    return socket.inet_ntoa(value[:4]), struct.unpack_from("<H", value, 4)[0]


# -- data packets ----------------------------------------------------------------
def data_packet(data_type: int, timestamp_ns: int, time_interval: int, dot_num: int,
                udp_cnt: int, frame_cnt: int, payload: bytes) -> bytes:
    head = DATA_HEADER.pack(0, DATA_HEADER.size + len(payload), time_interval, dot_num,
                            udp_cnt & 0xFFFF, frame_cnt & 0xFF, data_type, TIME_PTP, bytes(12),
                            0, timestamp_ns)
    return head + payload


def imu_packet(timestamp_ns: int, gyro, acc_mps2, udp_cnt: int = 0) -> bytes:
    acc_g = np.asarray(acc_mps2, dtype=np.float64) / G
    return data_packet(DATA_IMU, timestamp_ns, 0, 1, udp_cnt, 0,
                       IMU.pack(*map(float, gyro), *map(float, acc_g)))


def point_records(points_m: np.ndarray, reflectivity: int) -> np.ndarray:
    """(N, 3) metres -> the packed records; a miss (0, 0, 0) reflects nothing."""
    rec = np.zeros(points_m.shape[0], dtype=POINT)
    mm = np.round(points_m.astype(np.float64) * 1000.0).astype(np.int64)
    rec["x"], rec["y"], rec["z"] = mm[:, 0], mm[:, 1], mm[:, 2]
    hit = np.any(mm != 0, axis=1)
    rec["reflectivity"] = np.where(hit, reflectivity, 0)
    return rec


def time_interval(t_ns: np.ndarray) -> int:
    """The header's time_interval (0.1 us) for points at t_ns: the driver puts
    point i at timestamp + i * (time_interval * 100 // dot_num), so the span
    is the mean spacing times the count."""
    n = t_ns.shape[0]
    if n < 2:
        return 0
    spacing = (int(t_ns[-1]) - int(t_ns[0])) / (n - 1)
    return int(min(0xFFFF, max(0, round(spacing * n / 100.0))))


def packetize(rec: np.ndarray, t_ns: np.ndarray, per_packet: int) -> list[tuple[int, int, int, bytes]]:
    """Consecutive runs of at most per_packet points: (timestamp, time_interval,
    dot_num, payload) each."""
    out = []
    for a in range(0, rec.shape[0], per_packet):
        b = min(a + per_packet, rec.shape[0])
        out.append((int(t_ns[a]), time_interval(t_ns[a:b]), b - a, rec[a:b].tobytes()))
    return out


def split_at_boundaries(t_ns: np.ndarray, after_ns: int) -> tuple[list[slice], list[int]]:
    """Where a scan's points cross whole FRAME_NS: the runs between the
    boundaries and the boundaries in (after_ns, the last point]. The driver
    cuts a frame only when a packet ends on one, so each gets a packet of its
    own (`MARKER`) between the runs."""
    if t_ns.shape[0] == 0:
        return [], []
    # Never before the scan's own first point: on the first scan after_ns is
    # nothing yet, and from 0 the device clock's epoch is 1.7e10 boundaries.
    after_ns = max(after_ns, int(t_ns[0]) - 1)
    first = (after_ns // FRAME_NS + 1) * FRAME_NS
    bounds = list(range(first, int(t_ns[-1]) + 1, FRAME_NS))
    cuts = [0, *np.searchsorted(t_ns, bounds, side="left").tolist(), t_ns.shape[0]]
    return [slice(cuts[i], cuts[i + 1]) for i in range(len(cuts) - 1)], bounds


MARKER = np.zeros(1, dtype=POINT)  # a miss: dropped by every consumer's blind zone


# -- the device's clock ------------------------------------------------------------
class DeviceClock:
    """Sim seconds -> the device's nanoseconds, which must read as the host's
    epoch clock (module docstring). Tied once to the wall clock, then moved
    forward whenever the sim has fallen behind it by more than `slack_ns` —
    never back, since Point-LIO clears its buffers on a stamp that goes back.
    So a lag it has moved past must never be repaid: the main loop drops
    what one catch-up burst cannot cover and repays the rest before this
    reads the time (envs/pacing.py says what carrying it did)."""

    def __init__(self, slack_ns: int = 20_000_000, wall_ns=time.time_ns):
        self.slack_ns = slack_ns
        self._wall_ns = wall_ns
        self.offset_ns: int | None = None
        self.reanchors = 0

    def update(self, sim_time: float) -> None:
        lag = self._wall_ns() - int(sim_time * 1e9)
        if self.offset_ns is None:
            self.offset_ns = lag
        elif lag - self.offset_ns > self.slack_ns:
            self.offset_ns = lag
            self.reanchors += 1

    def ns(self, sim_time) -> np.ndarray | int:
        if self.offset_ns is None:
            raise RuntimeError("DeviceClock.update first")
        if np.ndim(sim_time):
            return self.offset_ns + np.round(np.asarray(sim_time) * 1e9).astype(np.int64)
        return self.offset_ns + int(round(sim_time * 1e9))


# -- the device ----------------------------------------------------------------------
@dataclass
class LivoxLinkConfig:
    lidar_ip: str = "192.168.1.172"  # the INI's [sensor.lidar] ip: the driver's lidar_configs
    dev_type: int = DEV_TYPES["mid360s"]  # the INI's [sensor.lidar] type
    serial: str = "SIMMID360S00001"
    points_per_packet: int = 96  # the MID360's own
    imu_hz: float = 200.0  # the MID360's own; Point-LIO warns past a 30 ms gap
    reflectivity: int = 100  # lidar.py: intensity is a constant; nothing reads it


class LivoxLink:
    """The device: answers discovery and commands, sends scans and IMU.

    Its sockets are bound to the lidar's own IP, never 0.0.0.0: the SDK on
    the same host binds host_ip on the same ports, drops anything sent from
    host_ip, and only knows a device by its source address and port. The
    broadcast is read on a socket bound to the broadcast address itself,
    which leaves unicast to the SDK's sockets alone."""

    def __init__(self, cfg: LivoxLinkConfig, log=print, clock: DeviceClock | None = None):
        self.cfg = cfg
        self.log = log
        self.clock = clock or DeviceClock()
        self.host: str | None = None
        self.point_target: tuple[str, int] | None = None
        self.imu_target: tuple[str, int] | None = None
        self.stats = {"discovery": 0, "commands": 0, "point_packets": 0, "imu_packets": 0,
                      "dropped": 0}
        self._seen: set[str] = set()
        self._udp_cnt = 0
        self._imu_cnt = 0
        self._frame_cnt = 0
        self._last_point_ns = -1
        self._next_imu = 0.0
        self.bcast = self._bind("255.255.255.255", PORT_DETECT)
        self.detect = self._bind(cfg.lidar_ip, PORT_DETECT)
        self.cmd = self._bind(cfg.lidar_ip, PORT_CMD)
        self.points = self._bind(cfg.lidar_ip, PORT_POINT)
        self.imu = self._bind(cfg.lidar_ip, PORT_IMU)
        self.log(f"[livox] a {'MID360s' if cfg.dev_type == 35 else 'MID360'} at "
                 f"{cfg.lidar_ip} (dev_type {cfg.dev_type}); waiting for the driver's search")

    @staticmethod
    def _bind(ip: str, port: int) -> socket.socket:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        try:
            s.bind((ip, port))
        except OSError as err:
            s.close()
            raise SystemExit(
                f"[livox] cannot bind {ip}:{port} ({err}). The lidar's IP has to be an "
                f"address of this host: README, 'The sim as the robot's lidar'") from err
        s.setblocking(False)
        return s

    def _once(self, key: str, line: str) -> None:
        if key not in self._seen:
            self._seen.add(key)
            self.log(line)

    # -- control ---------------------------------------------------------------
    def poll(self) -> None:
        for sock in (self.bcast, self.cmd):
            while True:
                try:
                    data, addr = sock.recvfrom(2048)
                except (BlockingIOError, InterruptedError):
                    break
                self.handle(data, addr, sock is self.bcast)

    def handle(self, data: bytes, addr: tuple[str, int], broadcast: bool) -> None:
        f = decode_frame(data)
        if f is None or f.cmd_type != CMD_TYPE_REQUEST:
            return
        if broadcast or f.cmd_id == CMD_DISCOVERY:
            if f.cmd_id != CMD_DISCOVERY:
                return
            self.stats["discovery"] += 1
            if self.host != addr[0]:
                self.host = addr[0]
                # Until the configuration names them, the SDK's fixed ports.
                self.point_target = self.point_target or (self.host, HOST_PORT_POINT)
                self.imu_target = self.imu_target or (self.host, HOST_PORT_IMU)
            self._once("found", f"[livox] found by the driver at {addr[0]}")
            ack = detection_payload(self.cfg.dev_type, self.cfg.serial, self.cfg.lidar_ip)
            self._send(self.detect, encode_frame(CMD_DISCOVERY, CMD_TYPE_ACK, f.seq, ack), addr)
            return
        self.stats["commands"] += 1
        if f.cmd_id == CMD_QUERY:
            ack = query_ack_payload(parse_query_keys(f.payload))
        elif f.cmd_id == CMD_PARAMS:
            kv = parse_kv(f.payload)
            point = host_target(kv.get(KEY_POINT_TARGET, b""))
            imu = host_target(kv.get(KEY_IMU_TARGET, b""))
            if point:
                self.point_target = point
            if imu:
                self.imu_target = imu
            if point or imu:
                self._once("configured", f"[livox] configured: points to {self.point_target}, "
                                         f"IMU to {self.imu_target}")
            ack = struct.pack("<BH", 0, 0)  # ret 0, error key 0: every value taken
        else:
            # Reboot, logs, upgrades: nothing the driver asks in a session.
            ack = struct.pack("<BH", 0, 0)
        self._send(self.cmd, encode_frame(f.cmd_id, CMD_TYPE_ACK, f.seq, ack), addr)

    def _send(self, sock: socket.socket, data: bytes, addr) -> None:
        try:
            sock.sendto(data, addr)
        except OSError as err:  # nobody listening yet is fine; anything else is worth a line
            self.stats["dropped"] += 1
            if not isinstance(err, ConnectionRefusedError):
                self._once(f"send:{err.errno}", f"[livox] send to {addr} failed: {err}")

    # -- data -------------------------------------------------------------------
    def tick(self, sim_time: float, read_imu) -> None:
        """The IMU at its rate, read from `read_imu()` -> (gyro rad/s, acc m/s^2)
        at the sim time it is called; a burst of catch-up steps gets one sample."""
        self.clock.update(sim_time)
        if self.imu_target is None or sim_time < self._next_imu:
            return
        period = 1.0 / self.cfg.imu_hz
        self._next_imu = max(self._next_imu + period, sim_time)
        gyro, acc = read_imu()
        self._imu_cnt += 1
        self._send(self.imu, imu_packet(self.clock.ns(sim_time), gyro, acc, self._imu_cnt),
                   self.imu_target)
        self.stats["imu_packets"] += 1

    def send_scan(self, points_m: np.ndarray, t_sim: np.ndarray) -> None:
        """One scan (Simulator.last_raw_scan): every ray in firing order, in the
        lidar frame at its firing, a miss as zeros."""
        if self.point_target is None or points_m.shape[0] == 0:
            return
        t_ns = self.clock.ns(t_sim)
        # A scan window can overlap the last by a physics step (the main loop
        # scans when due, not on a grid); the driver wants time to run forward.
        keep = t_ns > self._last_point_ns
        if not np.all(keep):
            points_m, t_ns = points_m[keep], t_ns[keep]
            if t_ns.shape[0] == 0:
                return
        rec = point_records(points_m, self.cfg.reflectivity)
        runs, bounds = split_at_boundaries(t_ns, self._last_point_ns)
        self._frame_cnt += 1
        for i, run in enumerate(runs):
            for ts, ti, n, payload in packetize(rec[run], t_ns[run], self.cfg.points_per_packet):
                self._send_points(ts, ti, n, payload)
            if i < len(bounds):
                self._send_points(bounds[i], 0, 1, MARKER.tobytes())
        self._last_point_ns = int(t_ns[-1])

    def _send_points(self, ts: int, ti: int, n: int, payload: bytes) -> None:
        self._udp_cnt += 1
        self._send(self.points, data_packet(DATA_CARTESIAN_HIGH, ts, ti, n, self._udp_cnt,
                                            self._frame_cnt, payload), self.point_target)
        self.stats["point_packets"] += 1

    def close(self) -> None:
        for s in (self.bcast, self.detect, self.cmd, self.points, self.imu):
            s.close()
