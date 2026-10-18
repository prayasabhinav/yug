import numpy as np

from yug import QuantileForecast
from yug.direction import CANT_SAY, DOWN, FLAT, UP, direction, read_key

LEVELS = [0.1, 0.5, 0.9]
KEY = {"margin": 1.0, "lower": 0.1, "upper": 0.9}


def band(low, mid, high):
    return np.array([low, mid, high], dtype=np.float32)


def same(got, want):
    return np.array_equal(got, np.array(want, dtype=np.float32), equal_nan=True)


def test_four_answers_against_the_last_value():
    # step 0: whole band above 10 + 1; step 1: whole band below 10 - 1;
    # step 2: band within the margin; step 3: band crossing the margin both ways
    values = band([12, 5, 9.5, 2], [13, 6, 10, 10], [14, 8, 10.5, 18])
    f = QuantileForecast(values, LEVELS)
    assert same(direction(f, 10.0, KEY), [UP, DOWN, FLAT, CANT_SAY])


def test_a_band_exactly_on_the_margin_is_flat():
    f = QuantileForecast(band([9], [10], [11]), LEVELS)
    assert same(direction(f, 10.0, KEY), [FLAT])


def test_a_band_crossing_one_edge_of_the_margin_is_cant_say():
    f = QuantileForecast(band([10.5], [12], [13]), LEVELS)
    assert same(direction(f, 10.0, KEY), [CANT_SAY])


def test_one_last_value_per_series():
    values = np.stack([band([12], [13], [14]), band([12], [13], [14])])
    f = QuantileForecast(values, LEVELS)
    assert same(direction(f, [10.0, 20.0], KEY), [[UP], [DOWN]])


def test_key_is_read_from_a_file(tmp_path):
    path = tmp_path / "key.json"
    path.write_text('{"margin": 0.5, "lower": 0.1, "upper": 0.9}')
    assert read_key(path) == {"margin": 0.5, "lower": 0.1, "upper": 0.9}
