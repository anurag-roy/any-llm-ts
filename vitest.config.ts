import { defineConfig } from "vitest/config";

export default defineConfig({
  test: {
    coverage: {
      exclude: ["src/index.ts", "src/types.ts"],
      provider: "v8",
      reporter: ["text", "json-summary"],
      thresholds: {
        // Native SDK adapters contain defensive fallbacks for several response
        // shapes that cannot all be produced by one installed SDK revision.
        branches: 80,
        functions: 85,
        lines: 85,
        statements: 85,
      },
    },
  },
});
