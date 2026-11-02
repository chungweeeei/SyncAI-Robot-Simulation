"""The observation and action plumbing of policy.py, against g2_deploy_rs, with
stand-in networks (no onnxruntime, no ONNX files)."""

import numpy as np

from syncai_sim.legged import pose_at_height
from syncai_sim.policy import (Q_DEFAULT, HimlocoRunner, MultiPolicy, PpoRunner,
                               StairHistoryRunner, projected_gravity)


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


def test_stair_history_lays_out_history_term_by_term():
    # Policy order = (HipX x4, HipY x4, Knee x4); u = 0 is the stance.
    action = np.array([1.0] * 4 + [0.0] * 4 + [-40.0] * 4, np.float32)
    net = Recorder(action)
    r = StairHistoryRunner(net)
    r.act(OMEGA, GRAV, [1.5, -0.2, 0.3], Q, DQ)
    first = net.seen[0]
    assert first.shape == (31 * 45,)
    r2p = [0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11]
    # Each term is its own block of 31 frames; the first call fills them all.
    assert np.allclose(first[0:93].reshape(31, 3), OMEGA * 0.25)
    assert np.allclose(first[93:186].reshape(31, 3), GRAV)
    assert np.allclose(first[186:279].reshape(31, 3), [1.5, -0.2, 0.3])  # not clipped
    assert np.allclose(first[279:651].reshape(31, 12), (Q - Q_DEFAULT)[r2p])
    assert np.allclose(first[651:1023].reshape(31, 12), (DQ * 0.05)[r2p])
    assert np.allclose(first[1023:], 0.0)
    target = r.act(OMEGA, GRAV, [0.9, 0.0, 0.0], Q, DQ)
    cmd = net.seen[1][186:279].reshape(31, 3)
    assert np.allclose(cmd[:30], [1.5, -0.2, 0.3]) and np.allclose(cmd[30], [0.9, 0.0, 0.0])
    last = net.seen[1][1023:].reshape(31, 12)
    assert np.allclose(last[:30], 0.0) and np.allclose(last[30], action)  # newest last
    # A*tanh(scale*u/A): small u ~ scale*u, large u saturates inside the soft limit.
    off = target - Q_DEFAULT
    assert np.allclose(off[[0, 3, 6, 9]], 0.125, atol=0.005)  # HipX, u = 1
    assert np.allclose(off[[1, 4, 7, 10]], 0.0)
    assert np.allclose(off[[2, 5, 8, 11]], -0.7647, atol=1e-3)  # Knee, u = -40
    assert (Q_DEFAULT[2] + off[2]) > 0.524  # above the URDF's knee limit
    r.on_enter()
    r.act(OMEGA, GRAV, [0, 0, 0], Q, DQ)
    assert np.allclose(net.seen[2][1023:], 0.0)


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
