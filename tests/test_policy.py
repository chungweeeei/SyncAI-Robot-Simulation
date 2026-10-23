"""The observation and action plumbing of policy.py, against g2_deploy_rs, with
stand-in networks (no onnxruntime, no ONNX files)."""

import numpy as np

from syncai_sim.legged import pose_at_height
from syncai_sim.policy import (Q_DEFAULT, HimlocoRunner, MultiPolicy, PpoRunner,
                               projected_gravity)


class Recorder:
    def __init__(self, action):
        self.action = np.asarray(action, np.float32)
        self.seen = []

    def __call__(self, obs):
        self.seen.append(np.array(obs, copy=True))
        return self.action


OMEGA = np.array([0.4, -0.8, 1.2])
GRAV = np.array([0.0, 0.0, -1.0], np.float32)
Q = Q_DEFAULT + np.arange(12, dtype=np.float32) * 0.01
DQ = np.arange(12, dtype=np.float32)


def test_projected_gravity_level_and_pitched():
    assert np.allclose(projected_gravity(np.eye(3)), [0, 0, -1])
    a = 0.3  # nose down about +y: gravity tips towards the body's +x
    R = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
    g = projected_gravity(R)
    assert g[0] > 0.29 and abs(g[1]) < 1e-9


def test_ppo_observation_layout_and_action():
    net = Recorder(np.full(12, 2.0))
    r = PpoRunner(net)
    target = r.act(OMEGA, GRAV, [1.5, -0.2, 0.3], Q, DQ)
    o = net.seen[0]
    assert o.shape == (45,)
    assert np.allclose(o[0:3], OMEGA * 0.25)
    assert np.allclose(o[3:6], GRAV)
    assert np.allclose(o[6:9], [1.0, -0.2, 0.3])  # clipped to +-1
    assert np.allclose(o[9:21], Q - Q_DEFAULT)
    assert np.allclose(o[21:33], DQ * 0.05)
    assert np.allclose(o[33:45], 0.0)  # no last action yet
    expected = np.clip(2.0 * PpoRunner.action_scale + Q_DEFAULT, PpoRunner.q_min, PpoRunner.q_max)
    assert np.allclose(target, expected)
    r.act(OMEGA, GRAV, [0, 0, 0], Q, DQ)
    assert np.allclose(net.seen[1][33:45], 2.0)  # the raw action comes back
    r.on_enter()
    r.act(OMEGA, GRAV, [0, 0, 0], Q, DQ)
    assert np.allclose(net.seen[2][33:45], 0.0)


def test_himloco_reorders_keeps_history_and_halves_hips():
    action = np.arange(12, dtype=np.float32)  # policy order
    net = Recorder(action)
    r = HimlocoRunner(net)
    r.act(OMEGA, GRAV, [0.5, 0.0, 0.0], Q, DQ)
    first = net.seen[0]
    assert first.shape == (270,)
    one = first[:45]
    assert np.allclose(one[0:3], [0.5, 0.0, 0.0])  # the command first, unclipped
    assert np.allclose(one[3:6], OMEGA * 0.25)
    assert np.allclose(one[6:9], GRAV)
    # Joints by type: hips (0, 3, 6, 9), thighs, calves.
    assert np.allclose(one[9:21], (Q - Q_DEFAULT)[[0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11]])
    assert np.allclose(first, np.tile(one, 6))  # the first call fills the history
    target = r.act(OMEGA, GRAV, [0.9, 0.0, 0.0], Q, DQ)
    second = net.seen[1]
    assert np.allclose(second[0:3], [0.9, 0.0, 0.0])  # newest first
    assert np.allclose(second[45:90], one[:33].tolist() + [0.0] * 12)
    assert np.allclose(second[33:45], action)  # last action, policy order
    robot = action[HimlocoRunner.policy2robot] * 0.25
    robot[[0, 3, 6, 9]] *= 0.5
    assert np.allclose(target, robot + Q_DEFAULT)


def test_multipolicy_falls_back_and_resets_on_switch():
    a, b = Recorder(np.zeros(12)), Recorder(np.zeros(12))
    ppo, him = PpoRunner(a), HimlocoRunner(b)
    mp = MultiPolicy({0: ppo, 1: him}, log=lambda *x: None)
    assert mp.select(3) is ppo  # IsaacSim / unknown -> PPO, as select_policy does
    ppo.last_action[:] = 1.0
    assert mp.select(0) is ppo and np.allclose(ppo.last_action, 1.0)  # same runner: kept
    assert mp.select(1) is him
    assert mp.select(0) is ppo and np.allclose(ppo.last_action, 0.0)  # re-entered: cleared


def test_pose_at_height_matches_the_leg_geometry():
    # Foot straight under the hip at h: thigh and shank close the triangle.
    for h in (0.12, 0.32):
        q = pose_at_height(h)
        hip, knee = q[1], q[2]
        z = 0.20 * np.cos(hip) + 0.21 * np.cos(hip + knee)
        x = 0.20 * np.sin(hip) + 0.21 * np.sin(hip + knee)
        assert abs(z - h) < 1e-9 and abs(x) < 1e-9
