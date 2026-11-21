import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from studio import brain

def run_tests():
    print("=== Testing Clip SEO Title, Hook & Description Generation ===")
    
    video_title = "I Survived 100 Hours In GTA 6"
    uploader = "MrBeast Gaming"
    
    # Synthetic words spanning 0 to 300 seconds
    sample_text = (
        "Yo look at this car right here! Bro this is crazy! "
        "Wait, did you see that explosion? He actually did it! "
        "No way this is real! How did we survive that? "
        "Look at the money dropping! I cannot believe what just happened! "
        "This is the most clutch moment ever! Watch out for the cops! "
        "We made it to the safehouse! Unreal!"
    )
    words = []
    t = 0.0
    for w in sample_text.split():
        words.append({"w": w, "s": round(t, 2), "e": round(t + 0.4, 2)})
        t += 0.5

    clips = [
        {"id": "clip01", "start": 0.0, "end": 15.0, "category": "highlight"},
        {"id": "clip02", "start": 15.0, "end": 35.0, "category": "shock"},
        {"id": "clip03", "start": 35.0, "end": 55.0, "category": "clutch"},
        {"id": "clip04", "start": 55.0, "end": 75.0, "category": "funny"},
        {"id": "clip05", "start": 75.0, "end": 95.0, "category": "fail"},
        {"id": "clip06", "start": 95.0, "end": 120.0, "category": "highlight"},
    ]

    titles = []
    hooks = []
    descriptions = []

    for idx, c in enumerate(clips, 1):
        seo = brain.generate_clip_seo(c, words, video_title, uploader, clip_index=idx, ai_enhance=False)
        print(f"\n--- Clip {idx} ({c['category']}) ---")
        print("TITLE:      ", seo["title"])
        print("HOOK:       ", seo["hook"])
        print("SCORE:      ", seo["score"])
        print("HASHTAGS:   ", seo["hashtags"])
        print("DESCRIPTION:\n", seo["description"])

        titles.append(seo["title"])
        hooks.append(seo["hook"])
        descriptions.append(seo["description"])

        # Assertions
        assert seo["title"], "Title must not be empty"
        assert seo["title"] not in ("Loud moment", "Clip"), "Title must not be generic"
        assert '""' not in seo["title"], f'Title must not contain empty quotes: {seo["title"]}'
        assert '()' not in seo["title"], f'Title must not contain empty parens: {seo["title"]}'
        assert seo["hook"], "Hook must not be empty"
        assert seo["description"], "Description must not be empty"
        assert "Timestamp:" in seo["description"], "Description must include timestamp"
        assert "#" in seo["description"], "Description must include hashtags"

    # Verify diversity across clips
    unique_titles = set(titles)
    unique_hooks = set(hooks)
    print(f"\nTotal clips: {len(clips)} | Unique titles: {len(unique_titles)} | Unique hooks: {len(unique_hooks)}")
    assert len(unique_titles) == len(clips), f"All titles must be unique! Found {len(unique_titles)} out of {len(clips)}"
    assert len(unique_hooks) >= len(clips) - 1, f"Hooks must be highly diverse! Found {len(unique_hooks)} out of {len(clips)}"

    # Test edge case: No words / empty transcript
    empty_clip = {"id": "clip99", "start": 200.0, "end": 230.0, "category": "highlight"}
    seo_empty = brain.generate_clip_seo(empty_clip, [], "Epic Minecraft Speedrun", "Dream", clip_index=99, ai_enhance=False)
    print("\n--- Clip without transcript words ---")
    print("TITLE: ", seo_empty["title"])
    print("HOOK:  ", seo_empty["hook"])
    assert seo_empty["title"], "Title must not be empty even without words"
    assert '""' not in seo_empty["title"], f'Title must not contain empty quotes: {seo_empty["title"]}'
    assert seo_empty["title"] not in ("Loud moment", "Clip")
    assert seo_empty["hook"]

    print("\nALL CLIP SEO TESTS PASSED!")

if __name__ == "__main__":
    run_tests()
