"""Tests for the trickiest, most silently-regressable logic in src/chunk.py:
the "Zero" boilerplate exclusion, the per-chunk entity cap, and the
paragraph-packing rule that chunks are never split mid-paragraph.
"""

from chunk import MAX_ENTITIES_PER_CHUNK, find_entities, paragraphs_to_chunks


class TestFindEntities:
    def test_matches_a_known_class(self):
        text = "Kaiser's Ascension skill now deals additional damage."
        assert find_entities(text) == ["Kaiser"]

    def test_case_sensitive_avoids_common_word_collision(self):
        # "will" as an ordinary auxiliary verb must not match the boss "Will".
        text = "This change will take effect in the next patch."
        assert find_entities(text) == []

    def test_capitalized_boss_name_still_matches(self):
        text = "Will's HP has been reduced by 10%."
        assert find_entities(text) == ["Will"]

    def test_zero_excluded_in_eligibility_boilerplate(self):
        text = "Zero characters that completed Story Quest Act 2 are eligible for this event."
        assert find_entities(text) == []

    def test_zero_kept_for_genuine_content(self):
        text = "Zero's Ego Weapon now deals additional damage against bosses."
        assert find_entities(text) == ["Zero"]

    def test_entities_under_cap_are_kept(self):
        text = "Hero, Paladin, and Bishop all received a skill balance update this patch."
        assert set(find_entities(text)) == {"Hero", "Paladin", "Bishop"}

    def test_over_cap_chunk_returns_no_entities(self):
        names = ["Hero", "Paladin", "Dark Knight", "Bishop", "Mercedes"]
        assert len(names) > MAX_ENTITIES_PER_CHUNK
        text = ", ".join(names) + " all received skill balance updates."
        assert find_entities(text) == []

    def test_no_known_entities_returns_empty_list(self):
        assert find_entities("General bug fixes and stability improvements.") == []


class TestParagraphsToChunks:
    def test_short_paragraphs_pack_into_one_chunk(self):
        paragraphs = ["short one", "short two"]
        assert paragraphs_to_chunks(paragraphs, max_chars=100) == ["short one\nshort two"]

    def test_splits_once_the_budget_is_exceeded(self):
        first = "a" * 50
        second = "b" * 60
        chunks = paragraphs_to_chunks([first, second], max_chars=100)
        assert chunks == [first, second]

    def test_single_oversized_paragraph_is_never_split_mid_paragraph(self):
        oversized = "x" * 150
        assert paragraphs_to_chunks([oversized], max_chars=100) == [oversized]

    def test_empty_input_returns_no_chunks(self):
        assert paragraphs_to_chunks([], max_chars=100) == []

    def test_packing_splits_before_entity_cap_is_exceeded(self):
        # Regression: table rows are one class/boss per paragraph (see
        # clean.py), so several short rows easily fit under max_chars
        # together -- but if each names a different entity, packing by
        # character budget alone can still assemble a chunk that exceeds
        # MAX_ENTITIES_PER_CHUNK, which then blanks entity tagging for the
        # whole chunk (including whatever real fact was in it).
        rows = [
            "Hero deals more damage now.",
            "Paladin deals more damage now.",
            "Dark Knight deals more damage now.",
            "Bishop deals more damage now.",
            "Mercedes deals more damage now.",
        ]
        chunks = paragraphs_to_chunks(rows, max_chars=1000)
        assert len(chunks) > 1  # character budget alone would keep these in one chunk
        for chunk_text in chunks:
            entities = find_entities(chunk_text)
            assert entities != []
            assert len(entities) <= MAX_ENTITIES_PER_CHUNK
