"""The sim as a MID360 (sensor/livox_link.py), against what Livox-SDK2 and
livox_ros_driver2 do with its bytes. The byte strings below are the SDK's own
(the workspace's src/third-party/Livox-SDK2, cited per test), so a test that
fails here is a device the real driver would not take."""

import binascii
import socket
import struct
import zlib

import numpy as np
import pytest

from syncai_sim.sensor import livox_link as ll


def sdk_request(cmd_id: int, seq: int, payload: bytes = b"") -> bytes:
    """A request as the SDK frames it (sdk_protocol.cpp PackLivoxLidarSdkPacket)."""
    return ll.encode_frame(cmd_id, ll.CMD_TYPE_REQUEST, seq, payload, sender=ll.SENDER_HOST)


def test_the_crcs_are_the_sdks():
    # FastCRC's check values for "123456789": CRC-16/CCITT-FALSE and CRC-32.
    assert binascii.crc_hqx(b"123456789", 0xFFFF) == 0x29B1
    assert zlib.crc32(b"123456789") == 0xCBF43926


def test_a_frame_round_trips_and_a_bad_crc_is_dropped():
    raw = ll.encode_frame(ll.CMD_PARAMS, ll.CMD_TYPE_ACK, 7, b"\x00\x00\x00")
    assert len(raw) == 27 and raw[0] == 0xAA
    f = ll.decode_frame(raw)
    assert (f.seq, f.cmd_id, f.cmd_type, f.payload) == (7, ll.CMD_PARAMS, 1, b"\x00\x00\x00")
    bad = bytearray(raw)
    bad[-1] ^= 1  # payload: the CRC-32 no longer matches
    assert ll.decode_frame(bytes(bad)) is None
    bad = bytearray(raw)
    bad[4] ^= 1  # header: the CRC-16 no longer matches
    assert ll.decode_frame(bytes(bad)) is None


def test_an_empty_payload_carries_a_zero_crc32():
    # sdk_protocol.cpp: crc32_d must be 0 with no data, or the frame is dropped.
    raw = sdk_request(ll.CMD_DISCOVERY, 1)
    assert len(raw) == 24 and struct.unpack_from("<I", raw, 20)[0] == 0
    assert ll.decode_frame(raw) is not None


def test_the_detection_answer_is_the_sdks_detection_data():
    p = ll.detection_payload(35, "SIMMID360S00001", "192.168.1.172")
    assert len(p) == 24  # define.h DetectionData
    assert p[0] == 0 and p[1] == 35  # ret, dev_type MID360s
    assert p[2:18].rstrip(b"\0") == b"SIMMID360S00001" and p[17] == 0  # a C string
    assert p[18:22] == bytes([192, 168, 1, 172])
    assert struct.unpack_from("<H", p, 22)[0] == 56100  # the SDK sends its config there


def test_the_firmware_query_is_answered_app_firmware():
    # general_command_handler.cpp QueryFwType: key_num 1, rsvd, key 0x8010.
    keys = ll.parse_query_keys(bytes.fromhex("010000001080"))
    assert keys == [ll.KEY_FW_TYPE]
    # fw_type 1: 0 is loader mode, and the configuration is never sent.
    assert ll.query_ack_payload(keys) == bytes.fromhex("0001001080010001")


def test_the_sdks_configuration_names_where_the_data_goes():
    # build_request.cpp BuildUpdateMid360LidarCfgRequest, host 192.168.1.103.
    host = bytes([192, 168, 1, 103])
    payload = struct.pack("<HH", 3, 0)
    for key, hp, lp in ((5, 56201, 56200), (6, 56301, 56300), (7, 56401, 56400)):
        payload += struct.pack("<HH", key, 8) + host + struct.pack("<HH", hp, lp)
    assert len(payload) == 40
    kv = ll.parse_kv(payload)
    assert ll.host_target(kv[ll.KEY_POINT_TARGET]) == ("192.168.1.103", 56301)
    assert ll.host_target(kv[ll.KEY_IMU_TARGET]) == ("192.168.1.103", 56401)


def test_a_point_packet_is_what_the_driver_reads():
    t = 1_700_000_000_000_000_000 + np.arange(96, dtype=np.int64) * 5000  # 5 us apart
    pts = np.zeros((96, 3))
    pts[:, 0] = 1.2346
    rec = ll.point_records(pts, 100)
    [(ts, ti, n, payload)] = ll.packetize(rec, t, 96)
    raw = ll.data_packet(ll.DATA_CARTESIAN_HIGH, ts, ti, n, 1, 0, payload)
    (version, length, time_interval, dot_num, _, _, data_type, time_type, _, _,
     stamp) = ll.DATA_HEADER.unpack_from(raw)
    assert length == len(raw) == 36 + 14 * 96  # the real lidar's 1380 bytes
    assert (version, dot_num, data_type, time_type) == (0, 96, 1, ll.TIME_PTP)
    assert stamp == t[0]
    # pub_handler.cpp: point i at timestamp + i * (time_interval * 100 // dot_num).
    step = time_interval * 100 // dot_num
    assert abs(step - 5000) <= 2
    x = np.frombuffer(raw[36:], dtype=ll.POINT)["x"]
    assert np.all(x == 1235)  # millimetres, rounded


def test_a_miss_goes_out_as_zeros_reflecting_nothing():
    rec = ll.point_records(np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]), 100)
    assert rec["reflectivity"].tolist() == [0, 100]


def test_an_imu_packet_is_gyro_then_acceleration_in_g():
    raw = ll.imu_packet(42, (0.1, 0.2, 0.3), (0.0, 0.0, 9.81))
    assert len(raw) == 60
    assert ll.DATA_HEADER.unpack_from(raw)[6] == ll.DATA_IMU
    vals = struct.unpack_from("<6f", raw, 36)
    assert vals[:3] == pytest.approx((0.1, 0.2, 0.3))
    assert vals[3:] == pytest.approx((0.0, 0.0, 1.0))  # Point-LIO multiplies by 9.81


def test_every_whole_100_ms_a_scan_spans_gets_a_packet_ending_on_it():
    # The driver (time_type 1) cuts a frame only on a packet whose last point
    # falls on a whole 100 ms; without the markers a gap at the boundary
    # would hold the frame for another 100 ms.
    t0 = 5 * ll.FRAME_NS - 30_000_000
    t = t0 + np.arange(0, 100_000_000, 50_000, dtype=np.int64)  # one scan, 50 us apart
    runs, bounds = ll.split_at_boundaries(t, t0 - 1)
    assert bounds == [5 * ll.FRAME_NS]
    assert t[runs[0]][-1] < bounds[0] <= t[runs[1]][0]
    assert sum(r.stop - r.start for r in runs) == t.shape[0]


def test_the_first_scan_looks_for_boundaries_in_itself_only():
    # Before any scan there is no last point: boundaries from the epoch would
    # be 1.7e10 of them (this ran the process out of memory once).
    t = 1_700_000_000_050_000_000 + np.arange(0, 100_000_000, 50_000, dtype=np.int64)
    runs, bounds = ll.split_at_boundaries(t, -1)
    assert bounds == [1_700_000_000_100_000_000]
    assert len(runs) == 2


def test_the_device_clock_reads_the_wall_clock_and_never_goes_back():
    wall = [10_000_000_000]
    clock = ll.DeviceClock(slack_ns=20_000_000, wall_ns=lambda: wall[0])
    clock.update(1.0)
    assert clock.ns(1.0) == wall[0]
    # The sim falls 50 ms behind the wall: moved forward, by the whole lag.
    wall[0] += 1_050_000_000
    clock.update(2.0)
    assert clock.ns(2.0) == wall[0] and clock.reanchors == 1
    # A sim ahead of the wall (a catch-up burst) is left ahead, not pulled back.
    before = clock.ns(3.0)
    clock.update(3.0)
    assert clock.ns(3.0) == before


@pytest.fixture
def link():
    """A device on 127.0.0.2 (every 127/8 address is this host's on Linux)."""
    try:
        dev = ll.LivoxLink(ll.LivoxLinkConfig(lidar_ip="127.0.0.2"), log=lambda *_: None)
    except SystemExit as err:
        pytest.skip(f"cannot bind the device's sockets here: {err}")
    yield dev
    dev.close()


def _host(port: int = 0) -> socket.socket:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(("127.0.0.1", port))
    s.settimeout(1.0)
    return s


def test_the_device_answers_the_search_and_the_configuration(link):
    host = _host()
    host_addr = host.getsockname()
    link.handle(sdk_request(ll.CMD_DISCOVERY, 3), host_addr, broadcast=True)
    data, src = host.recvfrom(2048)
    f = ll.decode_frame(data)
    # From the lidar's own IP and port 56000: the SDK keys the device by both.
    assert src == ("127.0.0.2", ll.PORT_DETECT)
    assert (f.cmd_type, f.seq, f.payload[1]) == (ll.CMD_TYPE_ACK, 3, 35)

    host.sendto(sdk_request(ll.CMD_QUERY, 4, bytes.fromhex("010000001080")),
                ("127.0.0.2", ll.PORT_CMD))
    payload = struct.pack("<HHHH", 1, 0, ll.KEY_POINT_TARGET, 8) + \
        socket.inet_aton("127.0.0.1") + struct.pack("<HH", host_addr[1], 56300)
    host.sendto(sdk_request(ll.CMD_PARAMS, 5, payload), ("127.0.0.2", ll.PORT_CMD))
    answers = {}
    for _ in range(50):
        link.poll()
        try:
            data, src = host.recvfrom(2048)
        except TimeoutError:
            break
        f = ll.decode_frame(data)
        answers[f.seq] = (f.cmd_id, f.payload, src)
        if len(answers) == 2:
            break
    assert answers[4][:2] == (ll.CMD_QUERY, bytes.fromhex("0001001080010001"))
    assert answers[5][:2] == (ll.CMD_PARAMS, b"\x00\x00\x00")
    assert answers[5][2] == ("127.0.0.2", ll.PORT_CMD)
    assert link.point_target == host_addr
    host.close()


def test_a_scan_streams_from_the_point_port_in_time_order(link):
    host = _host()
    link.point_target = host.getsockname()
    link.clock.update(0.0)
    t = np.linspace(0.0, 0.0099, 200)
    pts = np.tile([1.0, 0.0, 0.5], (200, 1))
    link.send_scan(pts, t)
    stamps, total = [], 0
    while True:
        try:
            data, src = host.recvfrom(4096)
        except TimeoutError:
            break
        assert src == ("127.0.0.2", ll.PORT_POINT)
        head = ll.DATA_HEADER.unpack_from(data)
        stamps.append(head[10])
        total += head[3]
        if total >= 200:
            break
    assert stamps == sorted(stamps)
    assert total >= 200  # the scan's points, plus a marker on any whole 100 ms
    host.close()
