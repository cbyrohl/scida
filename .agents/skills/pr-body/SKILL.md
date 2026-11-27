---
name: pr-body
description: Draft or revise scida pull request titles and descriptions, including stacked PRs. Use when preparing a scida PR for review or updating its description.
---

# Scida PR descriptions

Read the [contributor policy](../../../CONTRIBUTING.md#pull-requests),
[agent instructions](../../../AGENTS.md#ai-posting-on-github), and
[PR template](../../../.github/pull_request_template.md).

## Identify the diff

- For an existing PR, read its current body and inspect the actual base, head,
  and diff. Preserve relevant links, uploaded images, and artifacts. Keep the
  title unless the user requests a title change.
- For a prospective PR, use the intended base and head or the requested local
  worktree patch. Use the parent PR as the base for a stacked change. Clarify
  the base if two plausible choices change the scope.
- Prefer a short, outcome-oriented title. Describe the net change a reviewer
  will see against that base.

## Write the PR

1. **Why.** Start with the observed problem or user benefit. Link the issue or
   reproduction if it helps a reviewer understand the impact.
2. **What changed.** Describe observable behavior and important design choices.
   Mention relevant units, dtypes, scheduler behavior, and compatibility.
3. **Validation.** Lead with the strongest result that covers the current
   diff. Put the failing-then-passing regression or the measured performance
   result ahead of broad test counts. Give the reference or tolerance for
   numerical changes and enough setup to compare performance measurements.
   Include material failed, skipped, or pending checks and the revision covered
   by earlier results.

Use `Review notes` for decisions or limitations that need attention, `Stack`
for dependencies and review order, and `AI involvement` when assistance was
material. Put these after `Validation`. State the tool's role and the actual
human verification. Keep links usable from GitHub and leave out local absolute
paths and agent session links.

## Handoff

Return the proposed text for a local draft. For an authorized GitHub write,
verify the repository and account and follow the posting disclosure and
`agent-authored` rules. Change only the requested fields and read back the result.
