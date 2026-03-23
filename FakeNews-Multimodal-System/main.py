import argparse

from configs.config import TrainerConfig
from src.training.text_trainer import Model1ExpertTrainer, run_full_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FakeNews Multimodal System - Model 1 (Text)")
    parser.add_argument("--train", action="store_true", help="Train model from scratch")
    parser.add_argument("--predict", action="store_true", help="Run single-text inference")
    parser.add_argument("--test", action="store_true",
                        help="Run held-out test evaluation (evaluate_all). "
                             "Requires best_model.pt to exist.")
    parser.add_argument("--text", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.train:
        run_full_pipeline(TrainerConfig())
    elif args.predict:
        trainer = Model1ExpertTrainer(TrainerConfig())
        trainer.build_model()
        trainer.load_best_model()
        sample_text = args.text or "SHOCKING: Government Hides Truth About Aliens!"
        result = trainer.predict(sample_text)
        print("Text:", result["text"])
        print(f"Fake score  : {result['fake_score']:.4f} ({result['fake_class']})")
        print(f"Sentiment   : {result['sentiment_class']} (intensity={result['sentiment_intensity']:.2f})")
        print(f"Manipulation: {result['manipulation_score']:.4f} ({result['manipulation_class']})")
    elif args.test:
        trainer = Model1ExpertTrainer(TrainerConfig())
        trainer.load_data()          # builds test_loaders
        trainer.build_model()
        trainer.load_best_model()
        trainer.evaluate_all()
    else:
        print("No mode selected. Use --train, --predict, or --test.")


if __name__ == "__main__":
    main()
