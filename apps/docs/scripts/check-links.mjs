import { existsSync, globSync, readFileSync } from "node:fs";
import { join } from "node:path";

const outputRoot = new URL("../out/", import.meta.url);
const basePath = process.env.NEXT_PUBLIC_BASE_PATH ?? "";
const failures = [];

for (const file of globSync("**/*.html", { cwd: outputRoot })) {
  const html = readFileSync(new URL(file, outputRoot), "utf8");
  for (const match of html.matchAll(/href="(\/[^"]*)"/gu)) {
    const href = match[1].split(/[?#]/u)[0];
    if (href.length === 0) continue;
    const localHref =
      basePath.length > 0 && (href === basePath || href.startsWith(`${basePath}/`))
        ? href.slice(basePath.length) || "/"
        : href;

    const target = join(outputRoot.pathname, localHref);
    if (
      !existsSync(target) &&
      !existsSync(`${target}.html`) &&
      !existsSync(join(target, "index.html"))
    ) {
      failures.push(`${file}: ${href}`);
    }
  }
}

if (failures.length > 0) {
  console.error([...new Set(failures)].join("\n"));
  process.exitCode = 1;
} else {
  console.log("All rendered internal links resolve in the static export.");
}
