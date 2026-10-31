"""A small synthetic corpus with ground truth known BY CONSTRUCTION, covering 14 of the 15 benchmark categories.

Why synthetic: it lets the repository ship a benchmark whose answers, pages and evidence are verifiable offline
(`Dataset.validate_against_corpus`). It is a regression/ablation benchmark, NOT a claim about performance on real
documents: use `--dataset` / `--pdf-dir` to benchmark your own PDFs. The 15th category (OCR) needs scanned PDFs with
reference text; none are shipped, so OCR items are absent here (the CER metric itself is unit-tested).
"""
from __future__ import annotations

import random

from ..parsing.models import PageText
from .schema import BenchItem, Dataset, GTPage

V1, V2, HB = "orion_spec_v1.pdf", "orion_spec_v2.pdf", "harbor_report.pdf"

# ------------------------------------------------------------------ Orion-7 specification, revision 1
S = {
    "v1_maker": "The Orion-7 pump controller is manufactured by Kestrel Dynamics. Revision 1 of this specification was released in March 2021.",
    "v1_volt": "The nominal supply voltage of the Orion-7 in revision 1 is 24 volts DC.",
    "v1_amp": "The maximum supply current of the Orion-7 in revision 1 is 3.2 amperes.",
    "v1_power": "Electrical power is calculated as P = V x I, with V in volts and I in amperes.",
    "v1_t1": "| Mode | Flow rate (L/min) | Power (W) |\n| --- | --- | --- |\n| Eco | 12 | 35 |\n| Standard | 30 | 70 |\n| Turbo | 55 | 140 |",
    "v1_t2": "| Silent | 8 | 22 |\n| Boost | 70 | 190 |",
    "v1_temp": "The operating temperature range of the Orion-7 in revision 1 is -10 to 55 degrees Celsius.",
    "v1_derate": "Above 50 degrees Celsius the controller reduces the flow rate by 10 percent.",
    "v1_eff": "Efficiency is defined as flow rate in L/min divided by power in watts.",
    "v1_warr": "The warranty period for the Orion-7 in revision 1 is 36 months.",
    "v1_errata": "Erratum: the warranty period stated on the previous page is replaced by 24 months for units shipped after June 2022.",
    # ---- revision 2
    "v2_maker": "The Orion-7 pump controller is manufactured by Kestrel Dynamics. Revision 2 of this specification was released in January 2023.",
    "v2_volt": "The nominal supply voltage of the Orion-7 in revision 2 is 48 volts DC.",
    "v2_amp": "The maximum supply current of the Orion-7 in revision 2 is 2.1 amperes.",
    "v2_t1": "| Mode | Flow rate (L/min) | Power (W) |\n| --- | --- | --- |\n| Eco | 18 | 50 |\n| Standard | 45 | 90 |\n| Turbo | 80 | 170 |",
    "v2_t2": "| Silent | 10 | 30 |\n| Boost | 95 | 230 |",
    "v2_temp": "The operating temperature range of the Orion-7 in revision 2 is -20 to 60 degrees Celsius.",
    "v2_warr": "The warranty period for the Orion-7 in revision 2 is 36 months.",
    "v2_note": "Appendix note: laboratory measurements show a maximum supply current of 2.4 amperes for revision 2 units under load.",
    # ---- Port Velmara harbour report (needles)
    "hb_master": "The harbor master of Port Velmara is Ines Falk.",
    "hb_join": "Ines Falk joined the Velmara Port Authority in 2012 after ten years in the shipping industry.",
    "hb_ph1": "Phase one of the quay extension cost 2.1 million euros.",
    "hb_ph2": "Phase two of the quay extension cost 3.4 million euros.",
    "hb_appr": "The Marlow Quay crane replacement was approved in 2021.",
    "hb_done": "The Marlow Quay crane replacement was completed in 2023.",
    "hb_berth7": "Berth 7 is reserved for refrigerated cargo.",
    "hb_tab": "| Berth | Depth (m) |\n| --- | --- |\n| Berth 1 | 9.5 |\n| Berth 2 | 11.0 |\n| Berth 3 | 12.5 |\n| Berth 4 | 14.0 |",
}

_VESSELS = ["Aurelia", "Borealis", "Cormorant", "Daphne", "Estuary", "Fulmar", "Galatea", "Heron", "Ibis", "Juniper",
            "Kittiwake", "Lantern", "Mistral", "Nautilus", "Osprey", "Petrel", "Quillon", "Razorbill", "Skerry", "Tern"]
_CARGO = ["grain", "timber", "steel coils", "machinery", "fertiliser", "paper", "textiles", "scrap metal"]
_WEEK = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def _filler(rnd: random.Random, n: int = 6) -> str:
    out = []
    for _ in range(n):
        out.append(f"The vessel {rnd.choice(_VESSELS)} unloaded {rnd.choice(_CARGO)} on {rnd.choice(_WEEK)} "
                   f"and left the roads before nightfall.")
    return " ".join(out)


def build_corpus() -> dict[str, list[PageText]]:
    v1 = [S["v1_maker"], f"{S['v1_volt']} {S['v1_amp']} {S['v1_power']}", S["v1_t1"], S["v1_t2"],
          f"{S['v1_temp']} {S['v1_derate']} {S['v1_eff']}", S["v1_warr"], S["v1_errata"],
          "Appendix: dimensions and mounting. The housing is made of anodised aluminium."]
    v2 = [S["v2_maker"], f"{S['v2_volt']} {S['v2_amp']} {S['v1_power']}", S["v2_t1"], S["v2_t2"],
          f"{S['v2_temp']} {S['v1_derate']} {S['v1_eff']}", S["v2_warr"],
          "Appendix: dimensions and mounting. The housing is made of anodised aluminium.", S["v2_note"]]
    rnd = random.Random(1234)
    special = {4: S["hb_master"], 9: S["hb_ph1"], 12: S["hb_ph2"], 14: S["hb_tab"], 17: S["hb_appr"], 19: S["hb_join"],
               23: S["hb_done"], 27: S["hb_berth7"]}
    hb = []
    for p in range(1, 31):
        body = _filler(rnd, 5)
        hb.append(f"{special[p]}\n{body}" if p in special and special[p].startswith("|") else
                  f"{special[p]} {body}" if p in special else body)
    return {n: [PageText(i + 1, t) for i, t in enumerate(pages)]
            for n, pages in ((V1, v1), (V2, v2), (HB, hb))}


def corpus_texts(corpus: dict) -> dict[str, list[str]]:
    """{pdf: pages-or-ExtractedDoc} -> {pdf: [page text, ...]} for ground-truth validation."""
    return {k: [p.text for p in (v.pages if hasattr(v, "pages") else v)] for k, v in corpus.items()}


def _g(pdf, *pages):
    return [GTPage(pdf, p) for p in pages]


def build_dataset() -> Dataset:
    I = BenchItem
    items = [
        # ---- factual
        I("f1", "Who manufactures the Orion-7 pump controller?", "Kestrel Dynamics", "factual", "easy", _g(V1, 1), [S["v1_maker"]], aliases=["Kestrel Dynamics."]),
        I("f2", "What is the operating temperature range of the Orion-7 in revision 1?", "-10 to 55 degrees Celsius", "factual", "easy", _g(V1, 5), [S["v1_temp"]], aliases=["-10 to 55 °C"]),
        I("f3", "Which berth is reserved for refrigerated cargo at Port Velmara?", "Berth 7", "factual", "easy", _g(HB, 27), [S["hb_berth7"]], aliases=["7"]),
        # ---- multi-hop (two facts on different pages must be chained)
        I("m1", "In which year did the harbor master of Port Velmara join the port authority?", "2012", "multi_hop", "hard", _g(HB, 4, 19), [S["hb_master"], S["hb_join"]]),
        I("m2", "How many months passed between the release of revision 1 and the release of revision 2 of the Orion-7 specification?", "22 months", "multi_hop", "hard",
          _g(V1, 1) + _g(V2, 1), ["Revision 1 of this specification was released in March 2021.", "Revision 2 of this specification was released in January 2023."], aliases=["22"]),
        # ---- cross-page
        I("c1", "How many years passed between the approval and the completion of the Marlow Quay crane replacement?", "2 years", "cross_page", "medium", _g(HB, 17, 23), [S["hb_appr"], S["hb_done"]], aliases=["two years", "2"]),
        I("c2", "What is the Boost mode flow rate of the Orion-7 revision 1, and what happens to the flow rate above 50 degrees Celsius?", "70 L/min; above 50 degrees Celsius the flow rate is reduced by 10 percent", "cross_page", "hard", _g(V1, 4, 5), ["| Boost | 70 | 190 |", S["v1_derate"]]),
        # ---- table
        I("t1", "What is the power consumption of the Orion-7 revision 1 in Turbo mode?", "140 W", "table", "easy", _g(V1, 3), ["| Turbo | 55 | 140 |"], aliases=["140 watts", "140"]),
        I("t2", "What is the power consumption of the Orion-7 revision 1 in Boost mode?", "190 W", "table", "medium", _g(V1, 4), ["| Boost | 70 | 190 |"], aliases=["190 watts", "190"]),
        I("t3", "What is the depth of Berth 3 at Port Velmara?", "12.5 m", "table", "easy", _g(HB, 14), ["| Berth 3 | 12.5 |"], aliases=["12.5 metres", "12.5"]),
        # ---- numerical calculation
        I("n1", "What is the total cost of phase one and phase two of the quay extension?", "5.5 million euros", "numerical", "medium", _g(HB, 9, 12), [S["hb_ph1"], S["hb_ph2"]], aliases=["5.5"]),
        I("n2", "By how much does the Turbo mode flow rate exceed the Eco mode flow rate for the Orion-7 revision 1?", "43 L/min", "numerical", "medium", _g(V1, 3), ["| Eco | 12 | 35 |", "| Turbo | 55 | 140 |"], aliases=["43"]),
        # ---- mathematical reasoning (a formula from the text must be applied)
        I("a1", "Using the formula in the specification, what is the maximum electrical input power of the Orion-7 revision 1?", "76.8 W", "math", "hard", _g(V1, 2), [S["v1_volt"], S["v1_amp"], S["v1_power"]], aliases=["76.8 watts", "76.8"]),
        I("a2", "What is the efficiency of the Orion-7 revision 1 in Standard mode, using the definition in the specification?", "0.43 L/min per W", "math", "hard", _g(V1, 3, 5), ["| Standard | 30 | 70 |", S["v1_eff"]], aliases=["0.43", "0.4286"]),
        # ---- long-document needles
        I("l1", "In which year was the Marlow Quay crane replacement approved?", "2021", "long_document", "medium", _g(HB, 17), [S["hb_appr"]]),
        I("l2", "How much did phase two of the quay extension cost?", "3.4 million euros", "long_document", "medium", _g(HB, 12), [S["hb_ph2"]], aliases=["3.4"]),
        # ---- ambiguous (the question does not name a revision; a good answer covers both or asks which)
        I("b1", "What is the nominal supply voltage of the Orion-7?", "24 volts DC in revision 1 and 48 volts DC in revision 2", "ambiguous", "medium", _g(V1, 2) + _g(V2, 2), [S["v1_volt"], S["v2_volt"]]),
        I("b2", "What is the power consumption in Standard mode?", "70 W in revision 1 and 90 W in revision 2", "ambiguous", "medium", _g(V1, 3) + _g(V2, 3), ["| Standard | 30 | 70 |", "| Standard | 45 | 90 |"]),
        # ---- contradictory (the corpus disagrees with itself; a good answer reports both)
        I("k1", "What is the warranty period of the Orion-7 revision 1?", "36 months, but 24 months for units shipped after June 2022", "contradictory", "hard", _g(V1, 6, 7), [S["v1_warr"], S["v1_errata"]]),
        I("k2", "What is the maximum supply current of the Orion-7 revision 2?", "2.1 amperes in the specification, but laboratory measurements show 2.4 amperes", "contradictory", "hard", _g(V2, 2, 8), [S["v2_amp"], S["v2_note"]]),
        # ---- unanswerable
        I("u1", "What is the Bluetooth range of the Orion-7?", "", "unanswerable", "easy", [], [], answerable=False, forbidden_terms=["bluetooth"]),
        I("u2", "Who is the chief executive officer of Kestrel Dynamics?", "", "unanswerable", "easy", [], [], answerable=False, forbidden_terms=["chief executive", "ceo"]),
        I("u3", "How many containers did Port Velmara handle in 1999?", "", "unanswerable", "medium", [], [], answerable=False, forbidden_terms=["1999"]),
        # ---- citation accuracy (single fact; the cited page must be exactly right)
        I("z1", "What is the maximum supply current of the Orion-7 revision 2?", "2.1 amperes", "citation", "medium", _g(V2, 2), [S["v2_amp"]], aliases=["2.1 A", "2.1"]),
        I("z2", "How much did phase one of the quay extension cost?", "2.1 million euros", "citation", "medium", _g(HB, 9), [S["hb_ph1"]], aliases=["2.1"]),
        # ---- context retention (answerable only with the earlier turn)
        I("r1", "And what is its maximum supply current?", "3.2 amperes", "context_retention", "medium", _g(V1, 2), [S["v1_amp"]], aliases=["3.2 A", "3.2"],
          history=[("What is the nominal supply voltage of the Orion-7 revision 1?", "The nominal supply voltage of the Orion-7 revision 1 is 24 volts DC.")]),
        I("r2", "When did she join the port authority?", "2012", "context_retention", "medium", _g(HB, 19), [S["hb_join"]],
          history=[("Who is the harbor master of Port Velmara?", "The harbor master of Port Velmara is Ines Falk.")]),
        # ---- hallucination resistance (plausible, false-premise or tempting-but-absent)
        I("h1", "What is the IP67 dust and water ingress rating of the Orion-7?", "", "hallucination", "medium", [], [], answerable=False, forbidden_terms=["ip67", "ingress"]),
        I("h2", "What was the cost of the Marlow Quay crane replacement?", "", "hallucination", "hard", [], [], answerable=False, forbidden_terms=["crane replacement cost", "crane replacement was funded"]),
        # ---- comparison
        I("p1", "Compare the nominal supply voltage of the Orion-7 in revision 1 and revision 2.", "24 volts DC in revision 1 and 48 volts DC in revision 2", "comparison", "medium", _g(V1, 2) + _g(V2, 2), [S["v1_volt"], S["v2_volt"]]),
        I("p2", "How does the Standard mode flow rate differ between revision 1 and revision 2 of the Orion-7?", "30 L/min in revision 1 and 45 L/min in revision 2", "comparison", "medium", _g(V1, 3) + _g(V2, 3), ["| Standard | 30 | 70 |", "| Standard | 45 | 90 |"]),
        I("p3", "Which revision of the Orion-7 has the lower maximum supply current?", "revision 2 (2.1 amperes versus 3.2 amperes)", "comparison", "hard", _g(V1, 2) + _g(V2, 2), [S["v1_amp"], S["v2_amp"]]),
    ]
    # deterministic dev/test split (alternating inside each category) so configuration selection never sees test items
    seen: dict[str, int] = {}
    for it in items:
        n = seen.get(it.category, 0)
        it.split = "dev" if n % 2 == 0 else "test"
        seen[it.category] = n + 1
    return Dataset(items, "synthetic_v1")
