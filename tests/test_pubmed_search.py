"""Offline tests with hand-written PubMed-style XML (no network). Live check: python src/pubmed_search.py"""
import json
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import pubmed_search as ps  # noqa: E402

XML = b"""<?xml version="1.0"?>
<PubmedArticleSet>
 <PubmedArticle><MedlineCitation><PMID Version="1">111</PMID><Article>
  <Journal><Title>Test Journal</Title><JournalIssue><PubDate><Year>2022</Year></PubDate></JournalIssue></Journal>
  <ArticleTitle>Sulforaphane and <i>glucose</i> control</ArticleTitle>
  <Abstract><AbstractText Label="BACKGROUND">Part one.</AbstractText><AbstractText Label="RESULTS">Part two.</AbstractText></Abstract>
  <AuthorList><Author><LastName>Kim</LastName><ForeName>Jin</ForeName></Author><Author><CollectiveName>Study Group</CollectiveName></Author></AuthorList>
  <PublicationTypeList><PublicationType>Randomized Controlled Trial</PublicationType></PublicationTypeList>
 </Article></MedlineCitation></PubmedArticle>
 <PubmedArticle><MedlineCitation><PMID Version="1">222</PMID><Article>
  <Journal><Title>Other</Title><JournalIssue><PubDate><MedlineDate>2019 Jan-Feb</MedlineDate></PubDate></JournalIssue></Journal>
  <ArticleTitle>No abstract paper</ArticleTitle>
 </Article></MedlineCitation></PubmedArticle>
</PubmedArticleSet>"""


def fake_get(endpoint, params):
    if endpoint == "esearch.fcgi":
        return json.dumps({"esearchresult": {"count": "57", "idlist": ["222", "111"]}}).encode()
    return XML


class Parsing(unittest.TestCase):
    def test_fields_and_missing_values(self):
        a, b = ps.parse_articles(XML)
        self.assertEqual((a["pmid"], a["year"], a["journal"]), ("111", "2022", "Test Journal"))
        self.assertEqual(a["title"], "Sulforaphane and glucose control")
        self.assertEqual(a["authors"], ["Jin Kim", "Study Group"])
        self.assertEqual(a["abstract"], "BACKGROUND: Part one. RESULTS: Part two.")
        self.assertEqual(a["publication_types"], ["Randomized Controlled Trial"])
        self.assertEqual(a["url"], "https://pubmed.ncbi.nlm.nih.gov/111/")
        self.assertFalse(b["has_abstract"])
        self.assertEqual((b["abstract"], b["authors"], b["year"]), ("", [], "2019"))


class Search(unittest.TestCase):
    def test_success_keeps_esearch_order(self):
        with mock.patch.object(ps, "_get", fake_get):
            r = ps.search_pubmed("sulforaphane glucose", 2)
        self.assertIsNone(r["error"])
        self.assertEqual(r["total_found"], 57)
        self.assertEqual([a["pmid"] for a in r["articles"]], ["222", "111"])
        json.dumps(r)

    def test_no_results(self):
        with mock.patch.object(ps, "_get", lambda e, p: json.dumps({"esearchresult": {"count": "0", "idlist": []}}).encode()):
            r = ps.search_pubmed("zzzz nothing")
        self.assertEqual((r["articles"], r["error"]), ([], None))
        self.assertIn("No PubMed results", r["note"])

    def test_network_error_does_not_raise(self):
        with mock.patch.object(ps, "_get", side_effect=ConnectionError("offline")):
            r = ps.search_pubmed("apple polyphenols")
        self.assertIn("offline", r["error"])
        self.assertEqual(r["articles"], [])

    def test_bad_response_does_not_raise(self):
        with mock.patch.object(ps, "_get", return_value=b"<html>not json</html>"):
            self.assertIsNotNone(ps.search_pubmed("x y")["error"])

    def test_empty_query_and_limits(self):
        self.assertEqual(ps.search_pubmed("   ")["error"], "Empty query.")
        seen = {}
        def spy(endpoint, params):
            seen.update(params) if endpoint == "esearch.fcgi" else None
            return json.dumps({"esearchresult": {"count": "0", "idlist": []}}).encode()
        with mock.patch.object(ps, "_get", spy):
            ps.search_pubmed("a b", 999)
        self.assertEqual(seen["retmax"], ps.MAX_RESULTS_LIMIT)


if __name__ == "__main__":
    unittest.main()
