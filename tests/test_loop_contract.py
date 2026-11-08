import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from agents import coder  # noqa: E402


def test_system_prompt_carries_the_input_contract():
    assert "c[:, 51:54]" in coder.SYSTEM
    assert "c[:, None, 51:53]" in coder.SYSTEM


def test_last_error_is_shown_for_every_strategy():
    _, user = coder.build_prompt("tune_optimizer", "x=1", "skills", [], "RuntimeError: boom")
    assert "LAST ERROR" in user and "RuntimeError: boom" in user
    _, user = coder.build_prompt("tune_optimizer", "x=1", "skills", [], None)
    assert "LAST ERROR" not in user


def test_lesson_guidance_covers_every_class():
    assert set(coder.LESSON_GUIDANCE) == {"pass", "regression", "inconclusive", "heldout_regression", "crash"}
    assert "Make NO claim" in coder.LESSON_GUIDANCE["crash"]
