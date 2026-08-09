# Changesets

Every pull request that changes the published package should include a changeset:

```bash
npm run changeset
```

Choose the appropriate semantic version bump and write a concise user-facing summary. Documentation,
test-only, and repository-infrastructure changes do not need a changeset.

After changesets land on `main`, the release workflow opens or updates a version pull request. Merging
that pull request publishes the package, creates the Git tag, and creates the GitHub release.
