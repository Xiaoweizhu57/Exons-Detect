import pytest

torch = pytest.importorskip("torch")

from exons_detect.metrics import compute_hidden_delta, weighted_score


def test_compute_hidden_delta_returns_expected_shape():
    observer_hidden = tuple(torch.randn(2, 5, 4) for _ in range(6))
    performer_hidden = tuple(torch.randn(2, 5, 4) for _ in range(6))

    delta = compute_hidden_delta(
        observer_hidden_states=observer_hidden,
        performer_hidden_states=performer_hidden,
        tau=0.15,
        alpha=10.0,
        hidden_num=4,
    )

    assert delta.shape == (2, 4)
    assert torch.all(delta >= 0)


def test_weighted_score_returns_finite_scores():
    batch_size = 2
    seq_len = 5
    vocab_size = 7
    hidden_size = 4
    layer_count = 6

    input_ids = torch.tensor(
        [
            [1, 2, 3, 4, 0],
            [1, 3, 2, 4, 5],
        ]
    )
    attention_mask = torch.tensor(
        [
            [1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1],
        ]
    )
    observer_logits = torch.randn(batch_size, seq_len, vocab_size)
    performer_logits = torch.randn(batch_size, seq_len, vocab_size)
    observer_hidden = tuple(torch.randn(batch_size, seq_len, hidden_size) for _ in range(layer_count))
    performer_hidden = tuple(torch.randn(batch_size, seq_len, hidden_size) for _ in range(layer_count))

    scores = weighted_score(
        input_ids=input_ids,
        attention_mask=attention_mask,
        performer_logits=performer_logits,
        observer_logits=observer_logits,
        pad_token_id=0,
        observer_hidden_states=observer_hidden,
        performer_hidden_states=performer_hidden,
        tau=0.15,
        alpha=10.0,
        hidden_num=4,
    )

    assert scores.shape == (2,)
    assert torch.isfinite(scores).all()
