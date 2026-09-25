# bagof-things

Template repository for `bagof` Python projects.

This template includes:

- a `pyproject.toml` configured for a `bagof.things` package
- a `bagof` namespace package under `src/`
- reusable GitHub Actions for linting, testing, and publishing

The workflow wrappers intentionally track `bagofseeds/actions@main` so
template-generated repositories inherit shared CI updates without manually
refreshing pinned workflow SHAs.

When using the template, replace `things` with your project-specific package
name.
