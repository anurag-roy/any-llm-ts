import { createMDX } from "fumadocs-mdx/next";
import path from "node:path";
import { fileURLToPath } from "node:url";

const withMDX = createMDX();
const basePath = process.env.NEXT_PUBLIC_BASE_PATH ?? "";
const repositoryRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../..");

/** @type {import("next").NextConfig} */
const config = {
  basePath,
  output: "export",
  reactStrictMode: true,
  turbopack: {
    root: repositoryRoot,
  },
};

export default withMDX(config);
