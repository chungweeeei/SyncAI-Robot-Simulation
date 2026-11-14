"""The battery's BMS on CAN: the robot's Foxlink Jeff-8S3P, on `can0`.

The robot's fifth CAN bus carries no motor: `can0` is the battery pack's BMS
(500 kbit/s, extended frames), which the controller's hardware layer reads
(g2_deploy_rs src/hardware/bms.rs) and reports on. With `bms_link: can` the
sim is that BMS, on a vcan interface of the same name (scripts/vcan_up.sh),
speaking its protocol (Foxlink "Jeff-8S3P CAN Protocol" rev 2.0, in
g2_deploy_rs docs/datasheets): the frame's id is the item, the data is
big-endian.

The BMS says nothing on its own (protocol 1). It is asked:

    a frame with an item's id       answered with that id and the item's data
    0x87 Start (no data)            every item below is then sent cyclically
    0x86 Stop                       until this

The items ("2.1 Real-Time Data"), those the controller reads marked *:

    0x01..0x0E *  cell 1..14 voltage      u16 mV  (an 8S pack: 9..14 read 0)
    0x13 0x14 0x15 *  THM0..2             i16 0.1 C  (cell, cell, FET)
    0x18 *  internal temperature          i16 0.1 C
    0x1A *  pack voltage                  u16 mV
    0x1B    battery voltage               u16 mV
    0x39 *  system current                i32 mA, charging positive
    0x45 0x46  max, min cell voltage      u16 mV
    0x8B *  SW event 1, 2                 2 x u16 flags (0: nothing tripped)
    0x94 *  battery state                 u16: bit 0 discharging, 1 charging, 2 idle
    0xAB *  HW event 1, 2                 2 x u16 flags (0)
    0xCB *  SOC 0.1 %, SOH %, discharge time s, coulomb status   4 x u16

The other items of that table (the AFE's registers 0x5B..0x61, the MCU status
0x93, the RTC) and the logs, calibration and serial-number commands are not
answered: nothing in the sim stands behind them.

The battery behind the numbers is a model, and small. The charge is
integrated from the power the robot draws:

    P = idle_w + sum over the joints of ( tau dq + k tau^2 )
    I = -P / V_pack          SOC -= I dt / capacity

`tau dq` is the mechanical power (negative when a joint is braked: the J60's
driver returns it to the bus), `k tau^2` the winding's heat, k = 1.5 R / Kt^2
from the J60 manual's line resistance 0.54 ohm and joint torque constants
(0.8982 Nm/A for the hips' J60-6, 1.3473 for the knees' J60-10): 1.00 and
0.45 W/Nm^2. A cell's voltage is a straight line from 3.0 V empty to 4.2 V
full, all eight equal; the pack is their sum, with no sag under load.

Not in the protocol document and this module's choices: the cycle's period
(the whole table every `bms_period_ms`, 100 ms: the rate the controller
republishes at); that 0x87 and 0x86 are not acknowledged; the pack's
capacity (15 Ah: three 5 Ah cells in parallel) and the idle draw (40 W: the
computer, the lidar and twelve drivers); the temperatures (30 C, no thermal
model). Nothing trips: an empty pack reads 0 % and goes on supplying.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

import numpy as np

from syncai_sim.locomotion.can_link import FRAME_SIZE, open_bus, pack_frame, unpack_frame
from syncai_sim.sensor.proprioception import Sensors

CAN_EFF_FLAG = 0x80000000  # extended frame format: every frame of this BMS
CAN_RTR_ERR = 0x60000000
CAN_EFF_MASK = 0x1FFFFFFF

ID_STOP, ID_START = 0x86, 0x87
ID_CELL_BASE, MAX_CELLS = 0x01, 14
ID_THM = (0x13, 0x14, 0x15)
ID_INTERNAL_TEMP = 0x18
ID_PACK_VOLTAGE, ID_BATTERY_VOLTAGE = 0x1A, 0x1B
ID_SYS_CURRENT = 0x39
ID_MAX_CELL, ID_MIN_CELL = 0x45, 0x46
ID_SW_EVENT, ID_BATTERY_STATE, ID_HW_EVENT, ID_SOC_SOH = 0x8B, 0x94, 0xAB, 0xCB
STATE_DSG, STATE_CHG, STATE_IDLE = 1 << 0, 1 << 1, 1 << 2

CELLS = 8  # 8S3P
CELL_EMPTY_V, CELL_FULL_V = 3.0, 4.2
TEMPERATURE_C = 30.0  # every thermistor: there is no thermal model
# The winding's heat per joint, W / Nm^2: 1.5 R / Kt^2 (the module docstring).
HEAT_W_PER_NM2 = np.array([1.5 * 0.54 / 0.8982 ** 2, 1.5 * 0.54 / 0.8982 ** 2,
                           1.5 * 0.54 / 1.3473 ** 2] * 4)
# Below this the pack is neither charging nor discharging (the controller's
# own threshold when it has to guess the mode from the current).
IDLE_CURRENT_A = 0.1


@dataclass
class BmsConfig:
    interface: str = "can0"
    period_ms: float = 100.0  # the whole table, once started
    soc: float = 0.8  # at start, 0..1
    capacity_ah: float = 15.0
    idle_w: float = 40.0


class Bms:
    def __init__(self, cfg: BmsConfig | None = None, log=print, sock=None):
        """`sock`: a socket in place of the interface's (tests: anything with
        send / recv of 16-byte frames)."""
        self.cfg = cfg or BmsConfig()
        self.log = log
        c = self.cfg
        if not 0.0 <= c.soc <= 1.0 or c.capacity_ah <= 0 or c.period_ms <= 0:
            raise ValueError(f"bms: soc {c.soc} in 0..1, capacity {c.capacity_ah} Ah and "
                             f"period {c.period_ms} ms above 0")
        if sock is None:
            try:
                sock = open_bus(c.interface)
            except OSError as err:
                raise SystemExit(
                    f"bms_link can: cannot open {c.interface}: {err}. The interface must exist "
                    f"where the sim runs (scripts/vcan_up.sh makes it; in the container that "
                    f"needs SIM_NETWORK_MODE=host)") from err
        sock.setblocking(False)
        self.sock = sock
        self.soc = float(c.soc)
        self.current = 0.0  # A, charging positive
        self.power = c.idle_w  # W drawn
        self.discharge_s = 0.0
        self.looping = False
        self._t: float | None = None
        self._next = 0.0
        self.requests = 0
        self.sent = 0
        self.tx_errors = 0
        log(f"[bms] the pack's BMS on {c.interface}: {self.soc * 100:.0f} % of {c.capacity_ah:g} Ah, "
            f"silent until asked (0x87 starts the cycle, every {c.period_ms:g} ms)")

    # -- the battery ------------------------------------------------------------------
    def cell_voltage(self) -> float:
        return CELL_EMPTY_V + (CELL_FULL_V - CELL_EMPTY_V) * self.soc

    def pack_voltage(self) -> float:
        return CELLS * self.cell_voltage()

    def _drain(self, dt: float, sensors: Sensors) -> None:
        tau = np.asarray(sensors.tau, dtype=float)
        dq = np.asarray(sensors.dq, dtype=float)
        self.power = self.cfg.idle_w + float(tau @ dq + HEAT_W_PER_NM2 @ (tau * tau))
        self.current = -self.power / self.pack_voltage()
        self.soc = min(1.0, max(0.0, self.soc + self.current * dt / (self.cfg.capacity_ah * 3600.0)))
        if self.current < -IDLE_CURRENT_A:
            self.discharge_s += dt

    def state(self) -> int:
        if self.current < -IDLE_CURRENT_A:
            return STATE_DSG
        return STATE_CHG if self.current > IDLE_CURRENT_A else STATE_IDLE

    # -- the protocol -----------------------------------------------------------------
    def item(self, iid: int) -> bytes | None:
        """An item's data, or None for an id this BMS does not answer."""
        mv = int(round(self.cell_voltage() * 1000.0))
        if ID_CELL_BASE <= iid < ID_CELL_BASE + MAX_CELLS:
            return struct.pack(">H", mv if iid - ID_CELL_BASE < CELLS else 0)
        if iid in ID_THM or iid == ID_INTERNAL_TEMP:
            return struct.pack(">h", int(round(TEMPERATURE_C * 10.0)))
        if iid in (ID_PACK_VOLTAGE, ID_BATTERY_VOLTAGE):
            return struct.pack(">H", min(0xFFFF, CELLS * mv))
        if iid == ID_SYS_CURRENT:
            return struct.pack(">i", int(round(self.current * 1000.0)))
        if iid in (ID_MAX_CELL, ID_MIN_CELL):
            return struct.pack(">H", mv)
        if iid in (ID_SW_EVENT, ID_HW_EVENT):
            return bytes(4)
        if iid == ID_BATTERY_STATE:
            return struct.pack(">H", self.state())
        if iid == ID_SOC_SOH:
            return struct.pack(">4H", int(round(self.soc * 1000.0)), 100,
                               int(self.discharge_s) & 0xFFFF, 0)
        return None

    def tick(self, t: float, sensors: Sensors) -> None:
        """Called every physics step at sim time t: the charge drawn since the
        last one, the requests answered, the cycle sent when it is due."""
        if self._t is not None and t > self._t:
            self._drain(t - self._t, sensors)
        self._t = t
        while True:
            try:
                frame = self.sock.recv(FRAME_SIZE)
            except OSError:  # nothing waiting, or the interface is down
                break
            if len(frame) < FRAME_SIZE:
                continue
            cid, _ = unpack_frame(frame)
            if not cid & CAN_EFF_FLAG or cid & CAN_RTR_ERR:
                continue
            iid = cid & CAN_EFF_MASK
            self.requests += 1
            if iid == ID_START:
                if not self.looping:
                    self.log("[bms] 0x87: the cycle starts")
                self.looping, self._next = True, t
            elif iid == ID_STOP:
                self.looping = False
            else:
                self._send(iid)
        if self.looping and t >= self._next:
            self._next = t + self.cfg.period_ms / 1000.0
            for iid in TABLE:
                self._send(iid)

    def _send(self, iid: int) -> None:
        data = self.item(iid)
        if data is None:
            return
        try:
            self.sock.send(pack_frame(iid | CAN_EFF_FLAG, data))
            self.sent += 1
        except OSError:  # the bus is down or its queue is full
            self.tx_errors += 1

    def stats(self) -> dict:
        return {"soc": round(self.soc * 100.0, 2), "voltage": round(self.pack_voltage(), 2),
                "current": round(self.current, 2), "looping": self.looping,
                "requests": self.requests, "sent": self.sent, "tx_errors": self.tx_errors}

    def close(self) -> None:
        self.sock.close()


# The cycle 0x87 starts, in id order.
TABLE = (*range(ID_CELL_BASE, ID_CELL_BASE + MAX_CELLS), *ID_THM, ID_INTERNAL_TEMP,
         ID_PACK_VOLTAGE, ID_BATTERY_VOLTAGE, ID_SYS_CURRENT, ID_MAX_CELL, ID_MIN_CELL,
         ID_SW_EVENT, ID_BATTERY_STATE, ID_HW_EVENT, ID_SOC_SOH)
