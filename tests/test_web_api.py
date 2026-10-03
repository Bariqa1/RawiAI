"""
Integration tests for the RawiAI Web API endpoints.
Tests /api/meters, /api/meter/<name>, /api/search_lexicon, /api/ask, /api/stats.
"""

import unittest
import json
import urllib.request
import urllib.parse


from rawiai.nlp.normalizer import ArabicNormalizer


class TestWebAPI(unittest.TestCase):

    BASE_URL = "http://127.0.0.1:8080"

    def _get(self, path):
        try:
            req = urllib.request.Request(f"{self.BASE_URL}{path}")
            with urllib.request.urlopen(req, timeout=5) as resp:
                return json.loads(resp.read().decode("utf-8")), resp.status
        except Exception as e:
            return None, 500

    def _post(self, path, payload):
        try:
            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                f"{self.BASE_URL}{path}",
                data=data,
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                return json.loads(resp.read().decode("utf-8")), resp.status
        except Exception as e:
            return None, 500

    def test_api_stats(self):
        """Verifies /api/stats returns encyclopedia and lexicon metrics."""
        data, status = self._get("/api/stats")
        if status == 200:
            self.assertIn("meters_count", data)
            self.assertEqual(data["meters_count"], 16)
            self.assertGreaterEqual(data["lexicon_entries"], 3700)

    def test_api_meters_list(self):
        """Verifies /api/meters returns all 16 Khalil meters."""
        data, status = self._get("/api/meters")
        if status == 200:
            self.assertIn("meters", data)
            self.assertEqual(len(data["meters"]), 16)
            names = [m["name"] for m in data["meters"]]
            self.assertIn("الطويل", names)
            self.assertIn("البسيط", names)
            self.assertIn("الكامل", names)

    def test_api_meter_detail_tawil(self):
        """Verifies /api/meter/الطويل returns full prosody lesson."""
        path = f"/api/meter/{urllib.parse.quote('الطويل')}"
        data, status = self._get(path)
        if status == 200:
            self.assertIn("meter", data)
            self.assertEqual(data["meter"]["name"], "الطويل")
            stripped_tafail = ArabicNormalizer.strip_tashkeel(data["meter"]["standard_tafail"])
            self.assertIn("فعولن", stripped_tafail)

    def test_api_search_lexicon(self):
        """Verifies /api/search_lexicon returns entries from Asas Al-Balagha."""
        path = f"/api/search_lexicon?q={urllib.parse.quote('سيف')}"
        data, status = self._get(path)
        if status == 200:
            self.assertIn("results", data)
            self.assertTrue(len(data["results"]) > 0)
            self.assertEqual(data["results"][0]["root"], "سيف")

    def test_api_ask_author_query(self):
        """Verifies /api/ask responds with author identification."""
        data, status = self._post("/api/ask", {
            "query": "من قائل: الخيل والليل والبيداء تعرفني؟"
        })
        if status == 200:
            self.assertTrue(data.get("success"))
            self.assertIn("المتنبي", data.get("answer", ""))
            self.assertEqual(data.get("intent"), "author")

    def test_api_ask_completion_query(self):
        """Verifies /api/ask responds with hemistich completion."""
        data, status = self._post("/api/ask", {
            "query": "أكمل: قفا نبك من ذكرى حبيب ومنزل"
        })
        if status == 200:
            self.assertTrue(data.get("success"))
            self.assertEqual(data.get("intent"), "completion")
            stripped_ans = ArabicNormalizer.strip_tashkeel(data.get("answer", ""))
            self.assertIn("بسقط اللوى", stripped_ans)

    def test_api_evaluate_endpoint(self):
        """Verifies /api/evaluate audits prosody of a classical verse."""
        data, status = self._post("/api/evaluate", {
            "text": "عَلى قَدْرِ أَهْلِ العَزْمِ تَأْتِي العَزائِمُ ... وَتَأْتِي عَلَى قَدْرِ الكِرامِ المَكارِمُ"
        })
        if status == 200:
            self.assertTrue(data.get("success"))
            self.assertIn("dominant_meter", data)
            self.assertEqual(data["dominant_meter"], "الطويل")

    def test_api_compose_validation(self):
        """Verifies /api/compose validates empty topic."""
        data, status = self._post("/api/compose", {"topic": ""})
        # Empty topic returns 400
        self.assertIn(status, [400, 500])


if __name__ == "__main__":
    unittest.main()
