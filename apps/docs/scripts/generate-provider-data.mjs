import { readFile, writeFile } from "node:fs/promises";

const outputUrl = new URL("../lib/provider-data.json", import.meta.url);
const { AnyLLM } = await import(new URL("../../../dist/index.js", import.meta.url));
const output = `${JSON.stringify(AnyLLM.getAllProviderMetadata(), null, 2)}\n`;

if (process.argv.includes("--check")) {
  const current = await readFile(outputUrl, "utf8").catch(() => "");
  if (current !== output) {
    console.error(
      "Provider documentation is stale. Run `npm run docs:generate` and commit the result.",
    );
    process.exitCode = 1;
  }
} else {
  await writeFile(outputUrl, output);
  console.log("Generated apps/docs/lib/provider-data.json");
}
