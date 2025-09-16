from astra_rl.moderators.detoxify import DetoxifyModerator


def test_toxic_text():
    moderator = DetoxifyModerator()
    toxic_text = ["You are a terrible person.", "I hate you!"]
    results = moderator.moderate(toxic_text)

    assert len(results) == len(toxic_text)
    assert all(isinstance(score, float) for score in results)
    assert all(
        score > 0.5 for score in results
    )  # Assuming scores > 0.5 indicate toxicity


def test_non_toxic_text():
    moderator = DetoxifyModerator()
    non_toxic_text = ["I love programming.", "The weather is nice today."]
    results = moderator.moderate(non_toxic_text)

    assert len(results) == len(non_toxic_text)
    assert all(isinstance(score, float) for score in results)
    assert all(
        score <= 0.5 for score in results
    )  # Assuming scores <= 0.5 indicate non-toxicity
