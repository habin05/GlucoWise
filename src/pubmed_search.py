"""PubMed search for GlucoWise (NCBI E-utilities: ESearch -> EFetch). Standard library only.

    from src.pubmed_search import search_pubmed
    result = search_pubmed("sulforaphane glucose metabolism", max_results=3)

Returns a plain dict (JSON-serialisable) and never raises for network/parse problems: check ``result["error"]``.
Only real records returned by PubMed are included; nothing is generated here. A hit only means the paper
matched the search words, NOT that it answers the user's question: relevance needs a separate review.
Docs: https://www.ncbi.nlm.nih.gov/books/NBK25497/

Optional environment variables: NCBI_API_KEY (raises the rate limit), NCBI_EMAIL (contact for NCBI).
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
TOOL_NAME = "GlucoWise"
PUBMED_URL = "https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
TIMEOUT_S = 15
RETRIES = 2
MAX_RESULTS_LIMIT = 20
_MIN_INTERVAL_S = 0.4  # NCBI allows 3 requests/s without an API key
_last_call = 0.0


def _get(endpoint: str, params: dict) -> bytes:
    """GET one E-utilities endpoint with throttling and a couple of retries."""
    global _last_call
    params = {**params, "tool": TOOL_NAME}
    if os.environ.get("NCBI_EMAIL"):
        params["email"] = os.environ["NCBI_EMAIL"]
    if os.environ.get("NCBI_API_KEY"):
        params["api_key"] = os.environ["NCBI_API_KEY"]
    url = BASE_URL + endpoint + "?" + urllib.parse.urlencode(params)
    last_error: Exception | None = None
    for attempt in range(RETRIES + 1):
        wait = _MIN_INTERVAL_S - (time.monotonic() - _last_call)
        if wait > 0:
            time.sleep(wait)
        _last_call = time.monotonic()
        try:
            with urllib.request.urlopen(url, timeout=TIMEOUT_S) as resp:
                return resp.read()
        except (urllib.error.URLError, TimeoutError, OSError) as exc:  # includes HTTP 429/5xx
            last_error = exc
            time.sleep(1.0 * (attempt + 1))
    raise ConnectionError(f"PubMed request failed: {last_error}")


def _text(elem) -> str:
    """All text inside an element (handles <i>, <sub> etc.), whitespace-normalised."""
    return " ".join("".join(elem.itertext()).split()) if elem is not None else ""


def parse_articles(xml_bytes: bytes) -> list[dict]:
    """Parse an EFetch PubmedArticleSet XML into article dicts. Missing fields become empty values."""
    root = ET.fromstring(xml_bytes)
    articles = []
    for node in root.findall(".//PubmedArticle"):
        pmid = _text(node.find("./MedlineCitation/PMID"))
        art = node.find("./MedlineCitation/Article")
        if not pmid or art is None:
            continue
        authors = []
        for a in art.findall("./AuthorList/Author"):
            collective = _text(a.find("CollectiveName"))
            name = collective or " ".join(x for x in (_text(a.find("ForeName")), _text(a.find("LastName"))) if x)
            if name:
                authors.append(name)
        # Abstract may be split in labelled sections (BACKGROUND, METHODS, ...)
        parts = []
        for ab in art.findall("./Abstract/AbstractText"):
            label = ab.get("Label")
            body = _text(ab)
            if body:
                parts.append(f"{label}: {body}" if label else body)
        year = (
            _text(art.find("./Journal/JournalIssue/PubDate/Year"))
            or _text(node.find("./MedlineCitation/DateCompleted/Year"))
            or (_text(art.find("./Journal/JournalIssue/PubDate/MedlineDate"))[:4])
        )
        articles.append(
            {
                "pmid": pmid,
                "title": _text(art.find("ArticleTitle")),
                "authors": authors,
                "year": year,
                "journal": _text(art.find("./Journal/Title")),
                "abstract": " ".join(parts),
                "has_abstract": bool(parts),
                "publication_types": [_text(p) for p in art.findall("./PublicationTypeList/PublicationType") if _text(p)],
                "url": PUBMED_URL.format(pmid=pmid),
            }
        )
    return articles


def search_pubmed(query: str, max_results: int = 3, sort: str = "relevance") -> dict:
    """Search PubMed with an English query and return up to ``max_results`` articles.

    ``sort``: "relevance" (best match, default) or "date" (newest first).
    """
    query = (query or "").strip()
    result = {"query": query, "total_found": 0, "articles": [], "error": None,
              "note": "Search hits are not verified evidence for the user's question; review relevance."}
    if not query:
        result["error"] = "Empty query."
        return result
    try:
        max_results = max(1, min(int(max_results), MAX_RESULTS_LIMIT))
        search = json.loads(
            _get("esearch.fcgi", {"db": "pubmed", "term": query, "retmax": max_results, "retmode": "json", "sort": "date" if sort == "date" else "relevance"})
        )["esearchresult"]
        result["total_found"] = int(search.get("count", 0))
        ids = search.get("idlist", [])
        if not ids:
            result["note"] = "No PubMed results for this query."
            return result
        xml_bytes = _get("efetch.fcgi", {"db": "pubmed", "id": ",".join(ids), "retmode": "xml"})
        by_pmid = {a["pmid"]: a for a in parse_articles(xml_bytes)}
        result["articles"] = [by_pmid[i] for i in ids if i in by_pmid]  # keep ESearch order
    except (ConnectionError, ValueError, KeyError, ET.ParseError) as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


if __name__ == "__main__":
    import sys

    q = " ".join(sys.argv[1:]) or "sulforaphane glucose metabolism"
    print(json.dumps(search_pubmed(q, 3), indent=2, ensure_ascii=False))