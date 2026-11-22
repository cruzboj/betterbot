import numpy as np
import pandas as pd
import pytest

from analysis.classify import (cluster_kinds, dtw_to_references, leave_one_out, nearer_centre, nearer_cluster_per_word,
                               nearest_neighbour, share_creative)

FLAT = np.zeros(12)
WAVE = np.array([0., 5., 0., 5., 0., 5., 0., 5., 0., 5., 0., 5.])


def test_dtw_to_references_is_zero_for_an_identical_curve():
    table = dtw_to_references([WAVE], [FLAT, WAVE])
    assert table.shape == (1, 2) and table[0, 1] == pytest.approx(0) and table[0, 0] > 0


def test_nearest_neighbour_takes_the_kind_of_the_closest_reference():
    table = np.array([[3., 1., 2.], [0.5, 4., 9.]])
    kinds, nearest = nearest_neighbour(table, ["factual", "creative", "factual"])
    assert kinds == ["creative", "factual"] and nearest == [1, 0]


def test_leave_one_out_never_uses_the_text_itself():
    # every text is closest to itself (0); without itself, a's nearest is c, b's is a, c's is b
    table = np.array([[0., 2., 1.], [2., 0., 3.], [1., 3., 0.]])
    kinds, nearest = leave_one_out(table, ["factual", "creative", "creative"])
    assert nearest == [2, 0, 0] and kinds == ["creative", "factual", "factual"]


def test_cluster_kinds_names_each_cluster_after_its_majority():
    names = cluster_kinds(np.array([0, 0, 1, 1, 1]), ["factual", "factual", "creative", "creative", "factual"])
    assert names == {0: "factual", 1: "creative"}


def test_cluster_kinds_refuses_two_clusters_of_the_same_kind():
    with pytest.raises(ValueError):
        cluster_kinds(np.array([0, 1]), ["factual", "factual"])


def test_nearer_centre_gives_the_kind_of_the_closer_centre_text():
    table = np.array([[5., 1., 2.], [5., 9., 2.]])      # centres are references 0 (label 0) and 2 (label 1)
    kinds = nearer_centre(table, centres=[0, 2], kind_of_cluster={0: "factual", 1: "creative"})
    assert kinds == ["creative", "creative"]
    kinds = nearer_centre(np.array([[1., 0., 2.]]), centres=[0, 2], kind_of_cluster={0: "factual", 1: "creative"})
    assert kinds == ["factual"]


def test_nearer_cluster_per_word_is_empty_before_the_first_window():
    profile = pd.DataFrame({"to factual": [np.nan, 1., 3.], "to creative": [np.nan, 2., 1.]})
    assert list(nearer_cluster_per_word(profile)) == ["", "factual", "creative"]


def test_share_creative_counts_only_labelled_words_in_the_range():
    labels = np.array(["", "factual", "creative", "creative", "factual"])
    assert share_creative(labels, 0, 4) == pytest.approx(2 / 3)
    assert np.isnan(share_creative(labels, 0, 1))
