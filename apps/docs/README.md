# Documentation site

This workspace contains the statically exported Fumadocs site for `any-llm-ts`.

From the repository root, use Node.js 22 or newer and run:

```bash
npm run docs:dev
npm run docs:check
```

The production export is written to `apps/docs/out`.

## Deployment settings

- `NEXT_PUBLIC_SITE_URL` sets the canonical origin used by metadata and Open Graph images.
- `NEXT_PUBLIC_BASE_PATH` sets a path prefix for project-site hosting, for example
  `/any-llm-ts` on GitHub Pages.

The default build uses `/` as its base path and `http://localhost:3000` as its metadata origin so
the repository remains deployment-provider neutral.

## Generated references

Run `npm run docs:generate` after provider registry changes. It builds the package and regenerates
`lib/provider-data.json`; CI fails when the committed data is stale.
