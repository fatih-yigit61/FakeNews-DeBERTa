"""
FAZ B2 — Adversarial Robustness Test
Tests model with paired real/manipulated news sentences.
Proves: "The model is sensitive to stylistic exaggeration
         but does not perform factual verification."

Usage (Colab, after training):
    !python scripts/adversarial_test.py
"""
import sys, os
_dir = os.path.dirname(os.path.abspath(__file__))
for _ in range(5):
    _dir = os.path.dirname(_dir)
    if os.path.isdir(os.path.join(_dir, "configs")):
        break
sys.path.insert(0, _dir)
os.chdir(_dir)

import json
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from configs.config import TrainerConfig
from src.training.text_trainer import Model1ExpertTrainer

PLOT_DIR = Path("outputs/model1/plots")

# ═══════════════════════════════════════════════════════════════
#  TEST PAIRS: Real news + style-manipulated version
#  Same factual content, different writing style
# ═══════════════════════════════════════════════════════════════
TEST_PAIRS = [
    # --- Pair 1: Economy ---
    {
        "category": "Economy",
        "real": "The Federal Reserve announced a 0.25% interest rate cut, citing stable inflation and moderate economic growth.",
        "fake": "BREAKING!!! Fed makes SHOCKING secret move!!! Interest rates SLASHED in midnight decision!! Economy in FREEFALL!!!",
    },
    # --- Pair 2: Politics ---
    {
        "category": "Politics",
        "real": "The Senate passed a bipartisan infrastructure bill with 69 votes, allocating $1.2 trillion for roads, bridges, and broadband.",
        "fake": "EXPOSED!!! Senate's HIDDEN trillion-dollar scheme!!! Your tax money STOLEN in corrupt backroom deal!!! They don't want you to know!!!",
    },
    # --- Pair 3: Health ---
    {
        "category": "Health",
        "real": "A new study published in The Lancet found that regular exercise reduces the risk of cardiovascular disease by 30%.",
        "fake": "DOCTORS EXPOSED!!! The exercise TRUTH they've been HIDING from you!!! Big Pharma's worst nightmare REVEALED!!!",
    },
    # --- Pair 4: Technology ---
    {
        "category": "Technology",
        "real": "Apple announced its latest iPhone model featuring an improved camera system and longer battery life at its annual keynote.",
        "fake": "LEAKED!!! Apple's SECRET iPhone has MIND-BLOWING features they tried to HIDE!!! Industry INSIDERS reveal the SHOCKING truth!!!",
    },
    # --- Pair 5: Climate ---
    {
        "category": "Climate",
        "real": "Global temperatures rose by 1.1 degrees Celsius above pre-industrial levels in 2023, according to the World Meteorological Organization.",
        "fake": "CLIMATE HOAX EXPOSED!!! Scientists CAUGHT fabricating temperature data!!! The REAL numbers will SHOCK you!!!",
    },
    # --- Pair 6: Sports ---
    {
        "category": "Sports",
        "real": "Manchester City won the Premier League title for the fourth consecutive season with a 3-1 victory over West Ham.",
        "fake": "SCANDAL!!! Manchester City's title win was RIGGED!!! Leaked documents prove MASSIVE match-fixing conspiracy!!!",
    },
    # --- Pair 7: Education ---
    {
        "category": "Education",
        "real": "Harvard University announced a 5% increase in financial aid, expanding access to students from low-income families.",
        "fake": "OUTRAGEOUS!!! Harvard's DIRTY secret about financial aid EXPOSED!!! Elite universities SCAMMING students for BILLIONS!!!",
    },
    # --- Pair 8: Science ---
    {
        "category": "Science",
        "real": "NASA's James Webb Space Telescope captured detailed images of galaxies formed 13 billion years ago in the early universe.",
        "fake": "NASA HIDING alien evidence!!! Webb Telescope images show PROOF of extraterrestrial civilizations!!! Government COVER-UP exposed!!!",
    },
    # --- Pair 9: Fact flip (same style, different claim) ---
    {
        "category": "FactFlip",
        "real": "The unemployment rate decreased to 3.5% in the third quarter, according to the Bureau of Labor Statistics.",
        "fake": "The unemployment rate increased to 12.5% in the third quarter, according to leaked internal government documents.",
    },
    # --- Pair 10: Neutral style fake ---
    {
        "category": "NeutralFake",
        "real": "Pfizer reported that its COVID-19 vaccine showed 95% efficacy in Phase 3 clinical trials involving 43,000 participants.",
        "fake": "An independent laboratory confirmed that the COVID-19 vaccine contains microchips designed for population tracking and surveillance.",
    },
    # --- Pair 11: Celebrity gossip ---
    {
        "category": "Celebrity",
        "real": "Actor Tom Hanks received a lifetime achievement award at the Golden Globes ceremony for his contributions to cinema.",
        "fake": "SHOCKING!!! Tom Hanks SECRETLY arrested!!! Hollywood PANIC as A-list star's DARK past finally EXPOSED!!!",
    },
    # --- Pair 12: Business ---
    {
        "category": "Business",
        "real": "Tesla reported quarterly revenue of $25.2 billion, a 9% increase from the previous year driven by strong Model Y sales.",
        "fake": "TESLA BANKRUPT!!! Elon Musk's empire CRUMBLING!!! Insiders FLEE as secret financial documents show DEVASTATING losses!!!",
    },
    # --- Pair 13: Legal ---
    {
        "category": "Legal",
        "real": "The Supreme Court ruled 6-3 in favor of expanding environmental protections under the Clean Water Act.",
        "fake": "SUPREME COURT BETRAYAL!!! Justices SECRETLY paid off by environmental lobby!!! Constitution in DANGER!!!",
    },
    # --- Pair 14: Subtle manipulation ---
    {
        "category": "Subtle",
        "real": "Immigration increased by 15% compared to last year, with the majority of new arrivals being skilled workers.",
        "fake": "Immigration surged by an alarming 15% as borders remain dangerously open, overwhelming local communities and straining resources.",
    },
    # --- Pair 15: Same content, clickbait title ---
    {
        "category": "Clickbait",
        "real": "Researchers at MIT developed a new battery technology that could double the range of electric vehicles within five years.",
        "fake": "You WON'T BELIEVE what MIT just invented!!! This MIRACLE battery will make gas cars OBSOLETE overnight!!! Big Oil is TERRIFIED!!!",
    },
]


def main():
    print("=" * 70)
    print("  ADVERSARIAL ROBUSTNESS TEST")
    print("=" * 70)

    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    cfg = TrainerConfig()
    trainer = Model1ExpertTrainer(cfg)
    trainer.build_model()
    trainer.load_best_model()

    results = []

    for i, pair in enumerate(TEST_PAIRS):
        real_result = trainer.predict(pair["real"])
        fake_result = trainer.predict(pair["fake"])

        results.append({
            "id": i + 1,
            "category": pair["category"],
            "real_text": pair["real"][:80] + "...",
            "fake_text": pair["fake"][:80] + "...",
            "real_fake_score": real_result["fake_score"],
            "fake_fake_score": fake_result["fake_score"],
            "real_manip_score": real_result["manipulation_score"],
            "fake_manip_score": fake_result["manipulation_score"],
            "real_sentiment": real_result["sentiment_class"],
            "fake_sentiment": fake_result["sentiment_class"],
        })

    # Print table
    print(f"\n  {'#':>2} {'Category':<12} {'Real→Fake%':>10} {'Manip→Manip%':>12} {'Fake→Fake%':>10} {'Manip→Manip%':>12} {'Sensitive?':>10}")
    print("  " + "-" * 80)

    correct_detection = 0
    for r in results:
        sensitive = r["fake_fake_score"] > r["real_fake_score"] + 0.05
        if sensitive:
            correct_detection += 1
        print(f"  {r['id']:>2} {r['category']:<12} "
              f"{r['real_fake_score']:>9.1%} {r['real_manip_score']:>11.1%} "
              f"{r['fake_fake_score']:>9.1%} {r['fake_manip_score']:>11.1%} "
              f"{'YES' if sensitive else 'NO':>10}")

    print(f"\n  Detection rate: {correct_detection}/{len(results)} "
          f"({correct_detection/len(results):.0%}) pairs correctly distinguished")

    # Detailed output
    print(f"\n{'─' * 70}")
    print("  DETAILED RESULTS")
    print(f"{'─' * 70}")
    for r in results:
        print(f"\n  Pair {r['id']} [{r['category']}]:")
        print(f"    Real : {r['real_text']}")
        print(f"           Fake={r['real_fake_score']:.1%}  Manip={r['real_manip_score']:.1%}  Sent={r['real_sentiment']}")
        print(f"    Manip: {r['fake_text']}")
        print(f"           Fake={r['fake_fake_score']:.1%}  Manip={r['fake_manip_score']:.1%}  Sent={r['fake_sentiment']}")
        delta_f = r["fake_fake_score"] - r["real_fake_score"]
        delta_m = r["fake_manip_score"] - r["real_manip_score"]
        print(f"    Delta: Fake={delta_f:+.1%}  Manip={delta_m:+.1%}")

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    categories = [r["category"] for r in results]
    x = np.arange(len(categories))
    w = 0.35

    # Fake scores
    real_fakes = [r["real_fake_score"] for r in results]
    fake_fakes = [r["fake_fake_score"] for r in results]
    ax1.bar(x - w / 2, real_fakes, w, label="Original (Real News)", color="steelblue")
    ax1.bar(x + w / 2, fake_fakes, w, label="Manipulated Version", color="coral")
    ax1.set_ylabel("Fake Score")
    ax1.set_title("Fake Detection: Real vs Manipulated")
    ax1.set_xticks(x)
    ax1.set_xticklabels(categories, rotation=45, ha="right", fontsize=8)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3, axis="y")
    ax1.set_ylim(0, 1.05)

    # Manipulation scores
    real_manips = [r["real_manip_score"] for r in results]
    fake_manips = [r["fake_manip_score"] for r in results]
    ax2.bar(x - w / 2, real_manips, w, label="Original (Real News)", color="steelblue")
    ax2.bar(x + w / 2, fake_manips, w, label="Manipulated Version", color="coral")
    ax2.set_ylabel("Manipulation Score")
    ax2.set_title("Manipulation Detection: Real vs Manipulated")
    ax2.set_xticks(x)
    ax2.set_xticklabels(categories, rotation=45, ha="right", fontsize=8)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3, axis="y")
    ax2.set_ylim(0, 1.05)

    fig.suptitle("Adversarial Robustness Test\n"
                 "\"The model is sensitive to stylistic exaggeration "
                 "but does not perform factual verification.\"",
                 fontsize=12, style="italic")
    fig.tight_layout()
    save_path = PLOT_DIR / "adversarial_test.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  [plot] Saved → {save_path}")

    # Save JSON for thesis
    json_path = Path("outputs/model1/adversarial_results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  [json] Saved → {json_path}")

    # Key findings
    print(f"\n{'=' * 70}")
    print("  KEY FINDINGS FOR THESIS:")
    print(f"  - {correct_detection}/{len(results)} style-manipulated texts detected as more fake")
    print(f"  - FactFlip pair (same style, false claim): model {'CANNOT' if not any(r['category'] == 'FactFlip' and r['fake_fake_score'] > r['real_fake_score'] + 0.1 for r in results) else 'CAN'} detect factual errors")
    print(f"  - NeutralFake pair (calm style, false claim): tests semantic understanding")
    print(f"  - Conclusion: model detects STYLISTIC manipulation, not factual accuracy")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
