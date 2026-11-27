We welcome contributions to scida in the form of bug reports, feature requests, design proposals and more.

For information on how to contribute, please see the https://cbyrohl.github.io/scida/developer/.

## Pull requests

Use the [PR template](.github/pull_request_template.md). Keep **Why**,
**What changed**, and **Validation** in that order. Lead with the user-visible
problem and the strongest evidence from the revision under review. Keep each
PR focused. For stacked changes, name the base PR and the order for review or
merge.

## AI-assisted contributions

AI-assisted contributions are welcome. The person submitting a pull request is
responsible for the full change, including its tests and documentation.

- **Disclose material assistance under `AI involvement`.** Name the tool and
  describe its role in the investigation, design, code, tests, or documentation.
  Say what a person reviewed or verified. Mark human verification as pending
  when it is still open. Commit attribution trailers are optional.
- **Label mostly agent-written patches `agent-authored`.** Apply the label
  regardless of who opens the PR, and keep it after merge. If you lack labeling
  permission, disclose the authorship in the body and ask a maintainer to apply
  the label. Posting on someone's behalf has its own [disclosure rule](AGENTS.md#ai-posting-on-github).
- **Review the entire diff.** Verify proposed APIs, references, and dependency
  or licensing changes. Be able to explain the behavior and flag any effect on
  scientific results, data integrity, compatibility, or releases.
- **Report what you verified.** Include the tests or checks actually run and
  their results. Distinguish local or agent-run checks, CI results, and human
  verification. State relevant limits such as skipped real-data tests or
  unavailable cluster coverage. Identify evidence tied to an earlier revision.
- **Show the expected scientific result.** For changes to numerical results,
  units, selections, or data integrity, include a small reproduction with
  independently checked values. For bug fixes, show the failure before the fix
  and the expected result after it. Exercise a public entry point when practical
  and check units, dtypes, or behavior across Dask chunk layouts when relevant.
- **Measure performance claims.** Compare the same workload before and after.
  Give the dataset or fixture, scheduler, environment, and metric. Identify
  synthetic benchmarks and any limits on applying the result to production data.
- **Use inputs cleared for the tools involved.** Keep credentials and private
  or unpublished data out of AI prompts unless sharing has been authorized.
  Share evidence through repository-relative paths or approved links.
