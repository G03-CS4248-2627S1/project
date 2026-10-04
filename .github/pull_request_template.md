## What and why

<!-- What this PR does and why. Link the issue: Closes #<n> -->

## Results

<!-- For experiment PRs: run ids, what was compared, and the outcome against the seed spread (plan §15.3). Delete otherwise. -->

## Checklist (workflow.md §4.3)

- [ ] `uv run pytest` passes, including `tests/reference/`
- [ ] New training code passed a cluster smoke run (spec §9), or there is none
- [ ] Results in `results/<run_id>/` come from this PR's own commits
- [ ] No dev, Adversarial SQuAD or SQuAD-Shifts results (stage 5 final scoring only)
- [ ] Plan, spec (`implementation_details.md`) and `configs/base.yaml` still match the code
