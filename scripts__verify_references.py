"""Verify every citation against Crossref/DataCite and fetch OA PDFs.

For each reference the script
  1. resolves the DOI (Crossref, or DataCite for arXiv DOIs),
  2. compares the returned title with the declared title (token overlap),
  3. records whether a permissive license is advertised,
  4. optionally downloads an OA PDF and requires the file to start with %PDF-.

Anything that returns HTML is rejected and recorded; no reference is invented.

Outputs
  results/zh/references_verified.json   per-reference verification record
  logs/references.log                   human-readable log
  sources_zh/<key>.pdf                  only real PDFs
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT / "scripts"))
from _paths import build_root, logs_dir, results_dir, sources_dir  # noqa: E402

OUT = build_root()
SRC = sources_dir()
LOGS = logs_dir()
RESULTS = results_dir()

UA = "EchoCLIP-TA-paper/1.0 (academic verification; project group)"

REFERENCES = [
    {
        "key": "ouyang2020echonet",
        "doi": "10.1038/s41586-020-2145-8",
        "title": "Video-based AI for beat-to-beat assessment of cardiac function",
        "venue": "Nature",
        "year": 2020,
        "authors": "Ouyang D, He B, Ghorbani A, et al.",
        "pdf": [
            "https://www.nature.com/articles/s41586-020-2145-8.pdf",
        ],
    },
    {
        "key": "christensen2024echoclip",
        "doi": "10.1038/s41591-024-02959-y",
        "title": "Vision-language foundation model for echocardiogram interpretation",
        "venue": "Nature Medicine",
        "year": 2024,
        "authors": "Christensen M, Vukadinovic M, Yuan N, Ouyang D.",
        "pdf": [
            "https://www.nature.com/articles/s41591-024-02959-y.pdf",
        ],
    },
    {
        "key": "du2025cardiacclip",
        "doi": "10.1007/978-3-032-04971-1_5",
        "title": "CardiacCLIP: Video-Based CLIP Adaptation for LVEF Prediction in a Few-Shot Manner",
        "venue": "MICCAI 2025 (LNCS)",
        "year": 2025,
        "authors": "Du Y, Guo J, Li X.",
        "pdf": [],
    },
    {
        "key": "leclerc2019camus",
        "doi": "10.1109/TMI.2019.2900516",
        "title": "Deep Learning for Segmentation Using an Open Large-Scale Dataset in 2D Echocardiography",
        "venue": "IEEE Transactions on Medical Imaging",
        "year": 2019,
        "authors": "Leclerc S, Smistad E, Pedrosa J, et al.",
        "pdf": [
            "https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8643253",
        ],
    },
    {
        "key": "radford2021clip",
        "doi": "10.48550/arXiv.2103.00020",
        "title": "Learning Transferable Visual Models From Natural Language Supervision",
        "venue": "ICML 2021",
        "year": 2021,
        "authors": "Radford A, Kim JW, Hallacy C, et al.",
        "pdf": ["https://arxiv.org/pdf/2103.00020"],
    },
    {
        "key": "guo2017calibration",
        "doi": "10.48550/arXiv.1706.04599",
        "title": "On Calibration of Modern Neural Networks",
        "venue": "ICML 2017",
        "year": 2017,
        "authors": "Guo C, Pleiss G, Sun Y, Weinberger KQ.",
        "pdf": ["https://arxiv.org/pdf/1706.04599"],
    },
    {
        "key": "angelopoulos2023conformal",
        "doi": "10.1561/2200000101",
        "title": "Conformal Prediction: A Gentle Introduction",
        "venue": "Foundations and Trends in Machine Learning",
        "year": 2023,
        "authors": "Angelopoulos AN, Bates S.",
        "pdf": ["https://arxiv.org/pdf/2107.07511"],
    },
    {
        "key": "khosla2020supcon",
        "doi": "10.48550/arXiv.2004.11362",
        "title": "Supervised Contrastive Learning",
        "venue": "NeurIPS 2020",
        "year": 2020,
        "authors": "Khosla P, Teterwak P, Wang C, et al.",
        "pdf": ["https://arxiv.org/pdf/2004.11362"],
    },
    {
        "key": "vaswani2017attention",
        "doi": "10.48550/arXiv.1706.03762",
        "title": "Attention Is All You Need",
        "venue": "NeurIPS 2017",
        "year": 2017,
        "authors": "Vaswani A, Shazeer N, Parmar N, et al.",
        "pdf": ["https://arxiv.org/pdf/1706.03762"],
    },
    {
        "key": "liu2022convnext",
        "doi": "10.1109/CVPR52688.2022.01167",
        "title": "A ConvNet for the 2020s",
        "venue": "CVPR 2022",
        "year": 2022,
        "authors": "Liu Z, Mao H, Wu CY, et al.",
        "pdf": ["https://arxiv.org/pdf/2201.03545"],
    },
    {
        "key": "madani2018echo",
        "doi": "10.1038/s41746-018-0065-x",
        "title": "Deep echocardiography: data-efficient supervised and semi-supervised deep learning towards automated diagnosis of cardiac disease",
        "venue": "npj Digital Medicine",
        "year": 2018,
        "authors": "Madani A, Ong JR, Tibrewal A, Mofrad MRK.",
        "pdf": ["https://www.nature.com/articles/s41746-018-0065-x.pdf"],
    },
    {
        "key": "zhang2025echovisionfm",
        "doi": "10.1038/s41467-025-66340-4",
        "title": "Echo-Vision-FM: a pre-training and fine-tuning framework for echocardiogram video vision foundation model",
        "venue": "Nature Communications",
        "year": 2025,
        "authors": "Zhang Z, Wu Q, Ding S, et al.",
        "pdf": ["https://www.nature.com/articles/s41467-025-66340-4.pdf"],
    },
    {
        "key": "holste2024echo",
        "doi": "10.1038/s43856-024-00538-3",
        "title": "Efficient deep learning-based automated diagnosis from echocardiography with contrastive self-supervised learning",
        "venue": "Communications Medicine",
        "year": 2024,
        "authors": "Holste G, Oikonomou EK, Mortazavi BJ, et al.",
        "pdf": ["https://www.nature.com/articles/s43856-024-00538-3.pdf"],
    },
    {
        "key": "vukadinovic2024echoprime",
        "doi": "10.48550/arXiv.2410.09704",
        "title": "EchoPrime: A Multi-Video View-Informed Vision-Language Model for Comprehensive Echocardiography Interpretation",
        "venue": "arXiv",
        "year": 2024,
        "authors": "Vukadinovic M, Tang X, Yuan N, et al.",
        "pdf": ["https://arxiv.org/pdf/2410.09704"],
    },
]


def fetch_json(url: str, timeout: float = 30.0) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8", "replace"))


def crossref(doi: str) -> dict:
    url = "https://api.crossref.org/works/" + urllib.parse.quote(doi)
    try:
        m = fetch_json(url)["message"]
        return {
            "ok": True,
            "source": "crossref",
            "doi": m.get("DOI"),
            "title": (m.get("title") or [""])[0],
            "container": (m.get("container-title") or [""])[0],
            "year": (m.get("issued", {}).get("date-parts", [[None]])[0][0]),
            "authors": [
                f"{a.get('family','')} {a.get('given','')}".strip()
                for a in (m.get("author") or [])[:5]
            ],
            "volume": m.get("volume"),
            "issue": m.get("issue"),
            "page": m.get("page"),
            "article_number": m.get("article-number"),
            "type": m.get("type"),
            "cited_by": m.get("is-referenced-by-count"),
            "licenses": [l.get("URL") for l in (m.get("license") or [])],
        }
    except urllib.error.HTTPError as exc:
        return {"ok": False, "source": "crossref", "error": f"HTTP {exc.code}"}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "source": "crossref", "error": f"{type(exc).__name__}: {exc}"}


def datacite(doi: str) -> dict:
    url = "https://api.datacite.org/dois/" + urllib.parse.quote(doi)
    try:
        a = fetch_json(url)["data"]["attributes"]
        return {
            "ok": True,
            "source": "datacite",
            "doi": a.get("doi"),
            "title": (a.get("titles") or [{}])[0].get("title", ""),
            "container": a.get("publisher") or "",
            "year": a.get("publicationYear"),
            "authors": [c.get("name", "") for c in (a.get("creators") or [])[:5]],
            "type": (a.get("types") or {}).get("resourceTypeGeneral"),
        }
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "source": "datacite", "error": f"{type(exc).__name__}: {exc}"}


def download_pdf(urls: list, dest: Path) -> dict:
    """Return the first candidate that is genuinely a PDF (magic %PDF-)."""
    attempts = []
    for url in urls:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "application/pdf"})
            with urllib.request.urlopen(req, timeout=60) as r:
                head = r.read(8)
                if not head.startswith(b"%PDF"):
                    attempts.append({"url": url, "result": "not_pdf", "magic": head[:8].hex()})
                    continue
                body = head + r.read()
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(body)
            with open(dest, "rb") as fh:
                magic = fh.read(5)
            if magic != b"%PDF-":
                dest.unlink(missing_ok=True)
                attempts.append({"url": url, "result": "post_write_check_failed"})
                continue
            return {
                "ok": True,
                "url": url,
                "path": str(dest),
                "bytes": len(body),
                "magic": magic.decode("latin-1"),
                "attempts": attempts,
            }
        except Exception as exc:  # noqa: BLE001
            attempts.append({"url": url, "result": f"{type(exc).__name__}: {exc}"})
    return {"ok": False, "attempts": attempts}


def title_overlap(want: str, got: str) -> str:
    w = [t for t in want.lower().replace(":", " ").replace("-", " ").split() if len(t) > 3]
    g = got.lower()
    if not w:
        return "n/a"
    hit = sum(1 for t in w if t in g)
    return f"{hit}/{len(w)}"


def main() -> int:
    SRC.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    records = []
    log_lines = []
    for ref in REFERENCES:
        doi = ref["doi"]
        rec = {
            "key": ref["key"],
            "doi": doi,
            "declared_title": ref["title"],
            "declared_venue": ref["venue"],
            "declared_year": ref["year"],
            "authors_short": ref["authors"],
        }
        meta = datacite(doi) if doi.startswith("10.48550/") else crossref(doi)
        rec["registry"] = meta
        rec["title_match"] = title_overlap(ref["title"], meta.get("title") or "")
        oa = any("creativecommons" in (l or "") for l in (meta.get("licenses") or []))
        rec["oa_license_advertised"] = oa
        rec["pdf"] = {"ok": False, "attempts": [], "note": "no candidate"}
        if ref["pdf"]:
            rec["pdf"] = download_pdf(ref["pdf"], SRC / f"{ref['key']}.pdf")
        line = (
            f"{ref['key']:26s} {doi:34s} match={rec['title_match']:>5s} "
            f"oa={str(oa):5s} pdf={str(rec['pdf'].get('ok')):5s} "
            f"{str(meta.get('title'))[:60]}"
        )
        log_lines.append(line)
        print(line, flush=True)
        records.append(rec)
        time.sleep(0.3)

    (RESULTS / "references_verified.json").write_text(
        json.dumps(records, indent=1, ensure_ascii=False), encoding="utf-8"
    )
    (LOGS / "references.log").write_text("\n".join(log_lines), encoding="utf-8")
    ok = sum(1 for r in records if r["registry"].get("ok"))
    pdfs = sum(1 for r in records if r["pdf"].get("ok"))
    print(f"\n{ok}/{len(records)} DOIs resolved, {pdfs} PDFs downloaded")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
