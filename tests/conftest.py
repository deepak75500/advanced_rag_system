import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from localpdf.backends import stand_in_backends  # noqa: E402
from localpdf.config import RAGConfig  # noqa: E402
from localpdf.parsing.models import PageText  # noqa: E402
from localpdf.store import KnowledgeBase  # noqa: E402


@pytest.fixture
def backends():
    return stand_in_backends()


@pytest.fixture
def cfg():
    return RAGConfig()


@pytest.fixture
def kb(backends, cfg):
    k = KnowledgeBase(":memory:", backends.embedder, cfg)
    yield k
    k.close()


def pages(*texts):
    return [PageText(i + 1, t) for i, t in enumerate(texts)]


@pytest.fixture
def small_kb(kb):
    kb.ingest_pages("solar_farm.pdf", pages(
        "The Helios solar farm was commissioned in 2019. The Helios farm has a capacity of 240 megawatts.",
        "Maintenance of the Helios farm is performed every spring by the Orbit crew.",
        "Financing came from a green bond issued at a coupon of 3.5 percent."))
    kb.ingest_pages("wind_farm.pdf", pages(
        "The Zephyr wind farm has 48 turbines and a capacity of 180 megawatts.",
        "Zephyr was commissioned in 2021 and is operated by the Gale company."))
    return kb
