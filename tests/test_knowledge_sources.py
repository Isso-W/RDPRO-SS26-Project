from knowledge.sources import DOG_BREED_SOURCES


def test_fixed_sources_match_mcp_ingest_contract():
    assert len(DOG_BREED_SOURCES) >= 3
    for source in DOG_BREED_SOURCES:
        assert set(source) == {"source_name", "url", "source_type"}
        assert source["source_name"]
        assert source["url"].startswith("https://")
