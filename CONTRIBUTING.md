# Contributing

## Setup

Use Node.js 20 or newer.

```bash
npm install
npm run check
```

The full check includes ESLint, strict TypeScript compilation, Vitest coverage, and package builds.

## Documentation

The Fumadocs site lives in `apps/docs` and requires Node.js 22 or newer.

```bash
npm run docs:dev
npm run docs:check
```

Provider metadata in `apps/docs/lib/provider-data.json` is generated from the built package. After
changing the registry, refresh it with:

```bash
npm run docs:generate
```

Commit the generated file with the registry change. CI verifies that it is current.

## Provider changes

Keep provider-specific behavior in `src/providers`. Add a registry row only when a provider is genuinely OpenAI-compatible and needs configuration rather than request or response translation. Providers with custom authentication, parameter mappings, or wire formats should get a dedicated adapter.

Every change should include tests for successful requests, validation failures, provider errors, and streaming errors where applicable. Do not add API keys to the repository; integration tests should read conventional environment variables and skip cleanly when credentials are unavailable.

## Pull requests

Keep commits focused and use Conventional Commit subjects such as `feat:`, `fix:`, `docs:`, and `test:`. Describe which provider operations were tested with live credentials and which were covered only with SDK mocks.
