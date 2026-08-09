# Documentation site

This workspace contains the statically exported Fumadocs site for `any-llm-ts`.

The library and docs use separate lockfiles so Changesets can manage the publishable root package.
From the repository root, use Node.js 22 or newer and install both projects:

```bash
npm ci
npm ci --prefix apps/docs
```

Then run:

```bash
npm run docs:dev
npm run docs:check
```

The production export is written to `apps/docs/out`.

## Deployment settings

- `NEXT_PUBLIC_SITE_URL` sets the canonical origin used by metadata and Open Graph images.
- `NEXT_PUBLIC_BASE_PATH` sets a path prefix for project-site hosting, for example
  `/any-llm-ts` on GitHub Pages.

The default build uses `/` as its base path. On Vercel, metadata uses the automatically provided
production URL. Elsewhere, it uses `http://localhost:3000` unless `NEXT_PUBLIC_SITE_URL` is set.

## Deploying with Vercel Git

Import `anurag-roy/any-llm-ts` as a new Vercel project, then use these settings:

- Root Directory: `apps/docs`
- Node.js Version: 24.x
- Build settings: use the values committed in `apps/docs/vercel.json`

The committed Vercel configuration selects Next.js and clears custom build/output overrides so
Vercel's Next.js builder handles the static export from `out`.
The API reference build also reads the library's root `src/` and `tsconfig.json`; Vercel's Git build
checks out those files even though `apps/docs` is the project Root Directory. Do not set
`NEXT_PUBLIC_BASE_PATH` on Vercel.

After adding a custom domain, set `NEXT_PUBLIC_SITE_URL` to its full HTTPS origin for the Production
environment and redeploy. This keeps canonical metadata pinned to the custom domain.

## Generated references

Run `npm run docs:generate` after provider registry changes. It builds the package and regenerates
`lib/provider-data.json`; CI fails when the committed data is stale.
