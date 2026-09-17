from datetime import UTC
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from src.worker.doctranslator.glossary import ExtractedGlossaryTerm
from src.worker.services.glossary_service import GlossaryService


@pytest.fixture
def service():
    with patch("src.worker.services.glossary_service.get_storage_client"):
        return GlossaryService()


class TestGlossaryService:
    def test_get_local_glossary_path(self, service):
        path = service._get_local_glossary_path("test_domain")
        assert "test_domain" in str(path)

    def test_local_cache_fresh_not_exists(self, service):
        assert service._local_cache_fresh(Path("nonexistent")) is False

    def test_local_cache_fresh_exists(self, service):
        mock_file = MagicMock(spec=Path)
        mock_file.stat.return_value.st_mtime = datetime.now(UTC).timestamp()
        assert service._local_cache_fresh(mock_file) is True

    def test_load_local_glossary_json_exists(self, service):
        with patch.object(
            service, "_get_local_glossary_path", return_value=Path("fake.json")
        ):
            with patch.object(service, "_local_cache_fresh", return_value=True):
                with patch(
                    "src.worker.services.glossary_service.Path.exists",
                    return_value=True,
                ):
                    with patch(
                        "src.worker.services.glossary_service.Path.read_text",
                        return_value='{"terms": []}',
                    ):
                        res = service._load_local_glossary_json("domain")
                        assert res == {"terms": []}

    def test_load_local_glossary_json_invalid(self, service):
        with patch.object(
            service, "_get_local_glossary_path", return_value=Path("fake.json")
        ):
            with patch.object(service, "_local_cache_fresh", return_value=True):
                with patch(
                    "src.worker.services.glossary_service.Path.exists",
                    return_value=True,
                ):
                    with patch(
                        "src.worker.services.glossary_service.Path.read_text",
                        return_value="invalid json",
                    ):
                        assert service._load_local_glossary_json("domain") is None

    def test_load_domain_glossary_success(self, service):
        data = {
            "glossary": {
                "en": {
                    "terms": [{"source_term": "court", "translations": {"fr": "cour"}}]
                }
            }
        }
        with patch.object(service, "_download_glossary_json", return_value=data):
            res = service.load_domain_glossary(
                domain="domain", target_language_name="fr"
            )
            assert len(res) == 1
            assert res[0].entries[0].source == "court"

    def test_load_domain_glossary_failure(self, service):
        with patch.object(
            service, "_download_glossary_json", side_effect=Exception("Fail")
        ):
            assert (
                service.load_domain_glossary(domain="domain", target_language_name="fr")
                == []
            )

    def _multi_bucket_data(self) -> dict:
        return {
            "glossary": {
                "en": {
                    "terms": [{"source_term": "court", "translations": {"fr": "cour"}}]
                },
                "de": {
                    "terms": [
                        {
                            "source_term": "Vertrag",
                            "translations": {"fr": "contrat"},
                        }
                    ]
                },
                "ru": {
                    "terms": [
                        {
                            "source_term": "\u0434\u043e\u0433\u043e\u0432\u043e\u0440",
                            "translations": {"fr": "contrat"},
                        }
                    ]
                },
            }
        }

    def test_load_domain_glossary_without_filter_flattens_every_bucket(self, service):
        """The historical, unfiltered behaviour every existing caller
        relies on: source_languages=None (the default) means no bucket is
        excluded."""
        with patch.object(
            service, "_download_glossary_json", return_value=self._multi_bucket_data()
        ):
            res = service.load_domain_glossary(
                domain="domain", target_language_name="fr"
            )
        assert len(res) == 1
        sources = {entry.source for entry in res[0].entries}
        assert sources == {
            "court",
            "Vertrag",
            "\u0434\u043e\u0433\u043e\u0432\u043e\u0440",
        }

    def test_load_domain_glossary_filters_to_requested_languages(self, service):
        """A document that only contains English and German must not have
        the Russian bucket's terms eligible to literal-string-match it."""
        with patch.object(
            service, "_download_glossary_json", return_value=self._multi_bucket_data()
        ):
            res = service.load_domain_glossary(
                domain="domain",
                target_language_name="fr",
                source_languages=["en", "de"],
            )
        assert len(res) == 1
        sources = {entry.source for entry in res[0].entries}
        assert sources == {"court", "Vertrag"}

    def test_load_domain_glossary_empty_filter_result_returns_no_glossary(
        self, service
    ):
        with patch.object(
            service, "_download_glossary_json", return_value=self._multi_bucket_data()
        ):
            res = service.load_domain_glossary(
                domain="domain",
                target_language_name="fr",
                source_languages=["ja"],
            )
        assert res == []

    def _display_name_bucket_data(self) -> dict:
        """Mirrors the real shipped/seeded glossary assets (e.g.
        `.local-tmp/assets-cache/glossaries/legal.json`), which bucket terms
        under display names ("English", "French") and key `translations` by
        display name too -- unlike terms this service writes itself, which
        use canonical ISO codes throughout."""
        return {
            "glossary": {
                "English": {
                    "terms": [
                        {
                            "source_term": "force majeure",
                            "translations": {
                                "Spanish": "fuerza mayor",
                                "French": "force majeure",
                            },
                        }
                    ]
                },
                "French": {
                    "terms": [
                        {
                            "source_term": "résiliation",
                            "translations": {"English": "termination"},
                        }
                    ]
                },
            }
        }

    def test_load_domain_glossary_matches_display_name_buckets(self, service):
        """Regression: display-name bucket keys ("English") must still be
        found when the caller filters on canonical detected-language codes
        (["en"]) -- previously the raw, un-normalized comparison meant every
        such bucket was silently filtered out, yielding an empty glossary
        for every domain seeded with the shipped display-name assets."""
        with patch.object(
            service,
            "_download_glossary_json",
            return_value=self._display_name_bucket_data(),
        ):
            res = service.load_domain_glossary(
                domain="legal",
                target_language_name="es",
                source_languages=["en"],
            )
        assert len(res) == 1
        assert res[0].entries[0].source == "force majeure"
        assert res[0].entries[0].target == "fuerza mayor"

    def test_load_domain_glossary_matches_display_name_translation_keys(self, service):
        """The `translations` sub-keys are display names too in the shipped
        assets; a canonical-code `target_language_name` ("en") must still
        find a translation stored under "English"."""
        with patch.object(
            service,
            "_download_glossary_json",
            return_value=self._display_name_bucket_data(),
        ):
            res = service.load_domain_glossary(
                domain="legal",
                target_language_name="en",
                source_languages=["fr"],
            )
        assert len(res) == 1
        assert res[0].entries[0].source == "résiliation"
        assert res[0].entries[0].target == "termination"

    def test_load_domain_glossary_unrecognized_bucket_is_excluded_when_filtering(
        self, service, caplog
    ):
        """Normalization only gates a bucket when a language filter is
        actually requested -- with no filter, an unrecognized key is still
        included verbatim (see the flattening test above); only a
        language-scoped request needs to confirm what a bucket key means
        before deciding whether it belongs to the requested set."""
        data = {
            "glossary": {
                "not-a-language": {
                    "terms": [
                        {"source_term": "x", "translations": {"en": "y"}},
                    ]
                }
            }
        }
        with patch.object(service, "_download_glossary_json", return_value=data):
            with caplog.at_level("WARNING"):
                res = service.load_domain_glossary(
                    domain="domain",
                    target_language_name="en",
                    source_languages=["en"],
                )
        assert res == []
        assert "does not normalize to a known language" in caplog.text

    def test_merge_buckets_each_term_by_its_own_source_language(self, service):
        """One extraction batch over a mixed EN/FR document must land its
        terms in two different buckets, not one bucket named after
        whatever the job happened to declare."""
        data: dict = {"glossary": {}}
        terms = [
            ExtractedGlossaryTerm(
                source="indemnification", target="Freistellung", source_language="en"
            ),
            ExtractedGlossaryTerm(
                source="indemnisation", target="Freistellung", source_language="fr"
            ),
        ]
        merged, added = service._merge_terms_into_json(
            data, target_language_name="de", new_terms=terms
        )
        assert added == 2
        assert set(merged["glossary"]) == {"en", "fr"}
        assert merged["glossary"]["en"]["terms"][0]["source_term"] == "indemnification"
        assert merged["glossary"]["fr"]["terms"][0]["source_term"] == "indemnisation"

    def test_merge_does_not_overwrite_an_existing_translation(self, service):
        data = {
            "glossary": {
                "en": {
                    "terms": [
                        {
                            "source_term": "contract",
                            "translations": {"de": "Vertrag"},
                        }
                    ]
                }
            }
        }
        terms = [
            ExtractedGlossaryTerm(
                source="contract", target="WRONG", source_language="en"
            )
        ]
        merged, added = service._merge_terms_into_json(
            data, target_language_name="de", new_terms=terms
        )
        assert added == 0
        assert merged["glossary"]["en"]["terms"][0]["translations"]["de"] == "Vertrag"

    def test_merge_adds_a_new_target_language_to_an_existing_term(self, service):
        data = {
            "glossary": {
                "en": {
                    "terms": [
                        {
                            "source_term": "contract",
                            "translations": {"de": "Vertrag"},
                        }
                    ]
                }
            }
        }
        terms = [
            ExtractedGlossaryTerm(
                source="contract", target="contrat", source_language="en"
            )
        ]
        merged, added = service._merge_terms_into_json(
            data, target_language_name="fr", new_terms=terms
        )
        assert added == 1
        translations = merged["glossary"]["en"]["terms"][0]["translations"]
        assert translations == {"de": "Vertrag", "fr": "contrat"}

    def test_merge_drops_terms_with_no_source_language(self, service):
        """Defensive: callers are expected to filter these out first, but
        an ungrouped term must not silently land in an arbitrary bucket."""
        data: dict = {"glossary": {}}
        terms = [ExtractedGlossaryTerm(source="x", target="y", source_language="")]
        merged, added = service._merge_terms_into_json(
            data, target_language_name="de", new_terms=terms
        )
        assert added == 0
        assert merged["glossary"] == {}

    def test_merge_drops_a_plain_tuple_instead_of_raising(self, service, caplog):
        """Regression: a caller passing the old (source, target) 2-tuple
        shape (pre-dating per-term language attribution) must be dropped
        with a clear log line, not crash on `.source_language` access."""
        data: dict = {"glossary": {}}
        terms = [("contract", "Vertrag")]
        with caplog.at_level("WARNING"):
            merged, added = service._merge_terms_into_json(
                data, target_language_name="de", new_terms=terms
            )
        assert added == 0
        assert merged["glossary"] == {}
        assert "unexpected type tuple" in caplog.text

    def test_prefetch_domain_glossary(self, service):
        with patch.object(service, "_download_glossary_json") as mock_down:
            assert service.prefetch_domain_glossary("domain") is True
            mock_down.assert_called_once_with("domain", refresh=False)

    def test_download_glossary_json_success(self, service, tmp_path):
        with patch(
            "src.worker.services.glossary_service.get_storage_client"
        ) as mock_client:
            mock_blob = MagicMock()
            mock_blob.download_as_text.return_value = '{"terms": []}'
            mock_client.return_value.bucket.return_value.blob.return_value = mock_blob

            p = tmp_path / "gloss.json"
            with patch.object(service, "_get_local_glossary_path", return_value=p):
                res = service._download_glossary_json("domain", refresh=True)
                assert res == {"terms": []}
                assert p.exists()
