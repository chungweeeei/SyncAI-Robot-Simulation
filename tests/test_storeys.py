import numpy as np

from syncai_sim.storeys import storey_bounds, storey_of


def test_two_storeys_with_landings_between():
    rng = np.random.default_rng(0)
    first = rng.uniform(-1.0, -0.2, 2000)  # a tilted ground floor
    landings = np.concatenate([np.full(30, 1.1), np.full(20, 2.5), np.full(30, 3.8)])
    stairs = np.linspace(-0.2, 4.4, 15)
    second = rng.uniform(4.5, 5.4, 700)
    bounds = storey_bounds(np.concatenate([first, landings, stairs, second]))
    assert len(bounds) == 1
    # Under the upper floor, over the last landing: the flight goes with 1F.
    assert 3.8 < bounds[0] < 4.5
    assert storey_of([-0.5, 3.8, 5.0], bounds).tolist() == [0, 0, 1]


def test_one_storey_and_empty():
    assert storey_bounds(np.random.default_rng(1).uniform(-0.4, 0.0, 500)) == []
    assert storey_bounds(np.zeros(0)) == []
    assert storey_of([1.0, 99.0], []).tolist() == [0, 0]
