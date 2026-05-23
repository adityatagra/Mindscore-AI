"""
Master training script — trains all available modules sequentially.

Usage:
    python train_all.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config


def main():
    print("=" * 70)
    print("  MULTIMODAL MENTAL HEALTH SCORING SYSTEM - Full Training Pipeline")
    print("=" * 70)

    # Step 1: Generate dummy data if no real data exists
    need_gen = (
        not os.path.exists(config.TEXT_CONFIG["raw_csv"])
        or not os.path.exists(config.QUESTIONNAIRE_CONFIG["raw_csv"])
        or (config.AUDIO_CONFIG["enabled"] and not os.path.exists(config.AUDIO_CONFIG["processed_csv"]))
    )
    if need_gen:
        print("\n[INFO] Data not found. Generating datasets...")
        from src.data.generate_dummy_data import (
            generate_text_dataset, generate_questionnaire_dataset, generate_audio_feature_dataset,
        )
        if not os.path.exists(config.TEXT_CONFIG["raw_csv"]):
            generate_text_dataset()
        if not os.path.exists(config.QUESTIONNAIRE_CONFIG["raw_csv"]):
            generate_questionnaire_dataset()
        if config.AUDIO_CONFIG["enabled"] and not os.path.exists(config.AUDIO_CONFIG["processed_csv"]):
            generate_audio_feature_dataset()
        print()

    # Step 2: Train text module
    print("\n" + "-" * 70)
    from src.training.train_text_model import train as train_text
    train_text()

    # Step 3: Train questionnaire module
    print("\n" + "-" * 70)
    from src.training.train_questionnaire_model import train as train_quest
    train_quest()

    # Step 4: Train audio module (if enabled)
    if config.AUDIO_CONFIG["enabled"]:
        print("\n" + "-" * 70)
        from src.training.train_audio_model import train as train_audio
        train_audio()
    else:
        print("\n[INFO] Audio module is disabled. Skipping audio training.")

    print("\n" + "=" * 70)
    print("  Training complete! Models saved in:", config.MODELS_DIR)
    print("  Evaluations saved in:", config.OUTPUTS_DIR)
    print("=" * 70)
    print("\nNext step: Run the app with:")
    print("    streamlit run app/streamlit_app.py")


if __name__ == "__main__":
    main()
