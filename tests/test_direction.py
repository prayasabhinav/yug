import numpy as np

from yug import QuantileForecast
from yug.direction import DOWN, FLAT, UP, direction, read_key

LEVELS = [0.1, 0.5, 0.9]
KEY = {"margin": 1.0, "lower": 0.1, "upper": 0.9}


def band(low, mid, high):
    return np.array([low, mid, high], dtype=np.float32)


def test_up_down_flat_against_the_last_value():
    # step 0: whole band above 10 + 1; step 1: whole band below 10 - 1;
    # step 2: band inside the margin; step 3: band wider than the margin both ways
    values = band([12, 5, 9.5, 2], [13, 6, 10, 10], [14, 8, 10.5, 18])
    f = QuantileForecast(values, LEVELS)
    assert direction(f, 10.0, KEY).tolist() == [UP, DOWN, FLAT, FLAT]


def test_a_band_touching_the_margin_is_flat():
    f = QuantileForecast(band([11], [12], [13]), LEVELS)
    assert direction(f, 10.0, KEY).tolist() == [FLAT]


def test_one_last_value_per_series():
    values = np.stack([band([12], [13], [14]), band([12], [13], [14])])
    f = QuantileForecast(values, LEVELS)
    assert direction(f, [10.0, 20.0], KEY).tolist() == [[UP], [DOWN]]


def test_key_is_read_from_a_file(tmp_path):
    path = tmp_path / "key.json"
    path.write_text('{"margin": 0.5, "lower": 0.1, "upper": 0.9}')
    assert read_key(path) == {"margin": 0.5, "lower": 0.1, "upper": 0.9}
