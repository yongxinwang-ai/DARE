import torch

from verl.trainer.ppo.dllm_core_algos import (
    compute_policy_loss_bridgeratio,
    estimate_bridge_log_ratio,
    estimate_fisher_bridge_score,
    estimate_pseudolikelihood_log_ratio,
    estimate_thermobridge_log_ratio,
    select_bridge_alpha_by_ess,
)


def test_bridge_log_ratio_matches_coupled_logsumexp():
    current_paths = torch.tensor(
        [
            [[-1.0, -0.8], [-1.2, -0.6], [-0.9, -0.7]],
            [[-0.4, -1.1], [-0.3, -1.0], [-0.5, -0.9]],
        ]
    )
    old_paths = torch.tensor(
        [
            [[-1.1, -0.9], [-1.3, -0.5], [-1.0, -0.8]],
            [[-0.6, -1.0], [-0.4, -1.2], [-0.7, -0.8]],
        ]
    )

    log_ratio, metrics = estimate_bridge_log_ratio(current_paths, old_paths)
    expected = torch.logsumexp(current_paths, dim=1) - torch.logsumexp(old_paths, dim=1)

    assert torch.allclose(log_ratio, expected)
    assert metrics["bridge_ratio/num_paths"].item() == 3


def test_old_posterior_and_safebridge_limits():
    current_paths = torch.tensor([[[-1.0, -0.8], [-1.4, -0.7], [-0.9, -0.9]]])
    old_paths = torch.tensor([[[-1.2, -0.9], [-1.1, -0.8], [-1.0, -0.7]]])
    delta = current_paths - old_paths
    old_weights = torch.softmax(old_paths, dim=1)

    old_posterior_ratio, _ = estimate_bridge_log_ratio(current_paths, old_paths, estimator="old_posterior")
    exact_ratio = torch.logsumexp(current_paths, dim=1) - torch.logsumexp(old_paths, dim=1)
    assert torch.allclose(old_posterior_ratio, exact_ratio)

    first_order_ratio, _ = estimate_bridge_log_ratio(current_paths, old_paths, estimator="safebridge", alpha=0.0)
    expected_first_order = (old_weights * delta).sum(dim=1)
    assert torch.allclose(first_order_ratio, expected_first_order)


def test_safebridge_adaptive_alpha_reports_ess():
    current_paths = torch.tensor([[[-1.0, -0.8], [-1.4, -0.7], [-0.9, -0.9]]])
    old_paths = torch.tensor([[[-1.2, -0.9], [-1.1, -0.8], [-1.0, -0.7]]])

    selected_alpha, ess = select_bridge_alpha_by_ess(
        current_paths,
        old_paths,
        alpha_max=1.0,
        alpha_min=0.0,
        alpha_steps=5,
        ess_target=0.5,
    )

    assert selected_alpha.shape == current_paths[:, 0, :].shape
    assert torch.all((selected_alpha >= 0.0) & (selected_alpha <= 1.0))
    assert torch.all(ess > 0.0)


def test_cumulant_ratio_matches_second_order_formula():
    current_paths = torch.tensor([[[-1.0, -0.8], [-1.4, -0.7], [-0.9, -0.9]]])
    old_paths = torch.tensor([[[-1.2, -0.9], [-1.1, -0.8], [-1.0, -0.7]]])
    delta = current_paths - old_paths
    old_weights = torch.softmax(old_paths, dim=1)
    mean_delta = (old_weights * delta).sum(dim=1)
    var_delta = (old_weights * (delta - mean_delta.unsqueeze(1)).square()).sum(dim=1)

    log_ratio, _ = estimate_bridge_log_ratio(current_paths, old_paths, estimator="cumulant", alpha=0.5)
    expected = mean_delta + 0.25 * var_delta

    assert torch.allclose(log_ratio, expected)


def test_thermobridge_is_exact_for_constant_delta():
    old_paths = torch.tensor([[[-1.2, -0.9], [-1.1, -0.8], [-1.0, -0.7]]])
    current_paths = old_paths + 0.3

    log_ratio = estimate_thermobridge_log_ratio(current_paths, old_paths, num_points=7)

    assert torch.allclose(log_ratio, torch.full_like(log_ratio, 0.3), atol=1e-6)


def test_rao_blackwell_group_size_one_matches_bridge():
    current_paths = torch.tensor([[[-1.0, -0.8], [-1.4, -0.7], [-0.9, -0.9]]])
    old_paths = torch.tensor([[[-1.2, -0.9], [-1.1, -0.8], [-1.0, -0.7]]])

    rb_ratio, _ = estimate_bridge_log_ratio(current_paths, old_paths, estimator="rao_blackwell", rb_group_size=1)
    bridge_ratio, _ = estimate_bridge_log_ratio(current_paths, old_paths)

    assert torch.allclose(rb_ratio, bridge_ratio)


def test_fisher_bridge_score_is_posterior_weighted_path_score():
    path_scores = torch.tensor([[1.0, 2.0, 0.0]])

    score = estimate_fisher_bridge_score(path_scores)
    expected = (torch.softmax(path_scores, dim=1) * path_scores).sum(dim=1)

    assert torch.allclose(score, expected)


def test_pseudolikelihood_ratio_uses_token_conditional_delta():
    current_tokens = torch.tensor([[-0.9, -2.2]])
    old_tokens = torch.tensor([[-1.0, -2.0]])

    log_ratio = estimate_pseudolikelihood_log_ratio(current_tokens, old_tokens, scale=2.0)

    assert torch.allclose(log_ratio, 2.0 * (current_tokens - old_tokens))


def test_bridge_policy_loss_is_zero_for_zero_advantage():
    current_paths = torch.tensor([[[-1.0, -1.0], [-1.0, -1.0]]])
    old_paths = current_paths.clone()
    advantages = torch.zeros(1, 2)
    response_mask = torch.ones(1, 2)

    pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower, _ = compute_policy_loss_bridgeratio(
        old_l_theta_paths=old_paths,
        l_theta_paths=current_paths,
        advantages=advantages,
        response_mask=response_mask,
        cliprange=0.2,
    )

    assert torch.allclose(pg_loss, torch.tensor(0.0))
    assert torch.allclose(pg_clipfrac, torch.tensor(0.0))
    assert torch.allclose(ppo_kl, torch.tensor(0.0))
    assert torch.allclose(pg_clipfrac_lower, torch.tensor(0.0))
