import math
import random

import numpy as np
import pytest

from localpdf.bench import metrics as M
from localpdf.bench.resources import ResourceMonitor


def ref_lev(a, b):  # textbook DP
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[-1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def test_levenshtein_matches_reference_on_random_strings():
    rnd = random.Random(0)
    for _ in range(300):
        a = "".join(rnd.choice("abcd ") for _ in range(rnd.randint(0, 30)))
        b = "".join(rnd.choice("abcd ") for _ in range(rnd.randint(0, 30)))
        assert M.levenshtein(a, b) == ref_lev(a, b), (a, b)


def test_levenshtein_long_page_is_fast_and_correct():
    a = "the quick brown fox " * 200
    b = a.replace("quick", "quack", 40)
    assert M.levenshtein(a, b) == 40


def test_cer():
    assert M.character_error_rate("hello world", "hello world") == 0
    assert M.character_error_rate("hello world", "hallo world") == pytest.approx(1 / 11)
    assert math.isnan(M.character_error_rate("", "x"))
    assert M.character_error_rate("ab", "abcdef") == 2.0


def test_exact_match_and_f1():
    assert M.exact_match("The 24 volts.", "24 volts") == 1.0
    assert M.exact_match("about 24 volts", "24 volts") == 0.0
    assert M.exact_match("two years", "2 years", aliases=["two years"]) == 1.0
    assert M.contains_answer("It is about 24 volts DC", "24 volts") == 1.0
    assert M.token_f1("24 volts", "24 volts") == 1.0
    assert M.token_f1("", "24 volts") == 0.0
    assert M.token_f1("the supply is 24 volts dc", "24 volts") == pytest.approx(2 * (2 / 5) * 1 / (2 / 5 + 1))
    assert M.token_f1("six", "2 years", aliases=["six"]) == 1.0


G = {("a.pdf", 3), ("b.pdf", 1)}


def test_retrieval_metrics():
    ranked = [("c.pdf", 1), ("a.pdf", 3), ("x.pdf", 9), ("b.pdf", 1)]
    assert M.recall_at_k(ranked, G, 1) == 0 and M.recall_at_k(ranked, G, 2) == 1
    assert M.page_recall_at_k(ranked, G, 2) == 0.5 and M.page_recall_at_k(ranked, G, 4) == 1.0
    assert M.mrr(ranked, G) == 0.5
    assert M.ndcg_at_k([("a.pdf", 3), ("b.pdf", 1)], G, 5) == pytest.approx(1.0)
    assert 0 < M.ndcg_at_k(ranked, G, 4) < 1
    # a page retrieved twice must not be counted twice
    assert M.ndcg_at_k([("a.pdf", 3), ("a.pdf", 3), ("b.pdf", 1)], G, 3) <= 1.0
    assert M.context_precision([("a.pdf", 3), ("b.pdf", 1)], G) == 1.0
    assert M.context_precision(ranked, G) == pytest.approx((1 / 2 + 2 / 4) / 2)
    assert M.context_precision([("z", 1)], G) == 0.0
    assert math.isnan(M.mrr(ranked, set())) and math.isnan(M.recall_at_k(ranked, set(), 3))


def test_context_recall_and_faithfulness():
    assert M.context_recall(["Boost power is 190 watts"], ["The Boost mode draws 190 watts of power"]) == 1.0
    assert M.context_recall(["Boost power is 190 watts"], ["unrelated text"]) < 0.5
    assert M.faithfulness(["The farm has 48 turbines"], ["Zephyr farm has 48 turbines"]) == 1.0
    assert M.faithfulness(["Pirates built the farm"], ["Zephyr farm has 48 turbines"]) == 0.0
    assert math.isnan(M.faithfulness([], ["x"]))


def test_citation_metrics():
    assert M.citation_precision([("a.pdf", 3), ("q.pdf", 1)], G) == 0.5
    assert M.citation_recall([("a.pdf", 3)], G) == 0.5
    assert math.isnan(M.citation_precision([], G))
    assert M.citation_grounded("capacity of 240 megawatts", "The farm has a capacity of 240 megawatts.")
    assert not M.citation_grounded("capacity of 999 megawatts", "The farm has a capacity of 240 megawatts.")


def test_bootstrap_ci_is_reproducible_and_ignores_nan():
    xs = [1, 0, 1, 1, float("nan"), 0, 1]
    a, b = M.bootstrap_ci(xs), M.bootstrap_ci(xs)
    assert a == b and a[3] == 6 and a[1] <= a[0] <= a[2]
    assert M.nanmean([1, float("nan"), 3]) == 2


def test_resource_monitor_sees_allocation():
    with ResourceMonitor(interval=0.005) as m:
        blob = np.ones(30_000_000, dtype=np.float64)  # ~240 MB
        blob.sum()
        import time
        time.sleep(0.05)
    assert m.peak_ram_mb - m.baseline_ram_mb > 100
    assert m.as_dict()["peak_vram_mb"] is None or m.peak_vram_mb >= 0
