import { ArrowRight, Braces, PlugZap, RefreshCw } from "lucide-react";
import Link from "next/link";

const features = [
  {
    description: "Keep one application-facing API while changing the provider behind it.",
    icon: RefreshCw,
    title: "Switch providers",
  },
  {
    description: "Get typed parameters, results, streaming chunks, tools, and common errors.",
    icon: Braces,
    title: "Stay type-safe",
  },
  {
    description: "Use native adapters, registered providers, or your own OpenAI-compatible endpoint.",
    icon: PlugZap,
    title: "Extend when needed",
  },
];

export default function HomePage() {
  return (
    <main className="relative flex flex-1 flex-col overflow-hidden">
      <div className="hero-grid pointer-events-none absolute inset-0 -z-10" />
      <section className="mx-auto flex w-full max-w-6xl flex-col items-center px-6 pb-20 pt-24 text-center md:pt-32">
        <div className="mb-6 rounded-full border bg-fd-background/80 px-3 py-1 text-sm text-fd-muted-foreground backdrop-blur">
          Framework-independent · Promise-first · Apache-2.0
        </div>
        <h1 className="max-w-4xl text-balance text-5xl font-semibold tracking-tight md:text-7xl">
          One typed interface for your LLM providers
        </h1>
        <p className="mt-6 max-w-2xl text-balance text-lg text-fd-muted-foreground md:text-xl">
          Use chat completions, streaming, tools, embeddings, responses, images, moderation, and audio
          without coupling your application to one provider SDK.
        </p>
        <div className="mt-9 flex flex-wrap justify-center gap-3">
          <Link
            href="/docs"
            className="inline-flex h-11 items-center gap-2 rounded-lg bg-fd-primary px-5 font-medium text-fd-primary-foreground transition-opacity hover:opacity-90"
          >
            Get started <ArrowRight className="size-4" />
          </Link>
          <Link
            href="https://github.com/anurag-roy/any-llm-ts"
            className="inline-flex h-11 items-center rounded-lg border bg-fd-background px-5 font-medium transition-colors hover:bg-fd-accent"
          >
            View on GitHub
          </Link>
        </div>

        <div className="mt-14 w-full max-w-3xl overflow-hidden rounded-xl border bg-fd-card text-left shadow-2xl shadow-fd-primary/5">
          <div className="flex items-center gap-2 border-b px-4 py-3">
            <span className="size-2.5 rounded-full bg-red-400" />
            <span className="size-2.5 rounded-full bg-amber-400" />
            <span className="size-2.5 rounded-full bg-emerald-400" />
            <span className="ml-2 font-mono text-xs text-fd-muted-foreground">quickstart.ts</span>
          </div>
          <pre className="overflow-x-auto p-5 text-sm leading-7 md:p-7">
            <code>{`import { completion } from "any-llm-ts";

const response = await completion({
  provider: "openai",
  model: "gpt-4.1-mini",
  messages: [{ role: "user", content: "Hello!" }],
});

console.log(response.choices[0]?.message.content);`}</code>
          </pre>
        </div>
      </section>

      <section className="mx-auto grid w-full max-w-6xl gap-4 px-6 pb-24 md:grid-cols-3">
        {features.map(({ description, icon: Icon, title }) => (
          <div key={title} className="rounded-xl border bg-fd-card/70 p-6 backdrop-blur">
            <Icon className="mb-5 size-6 text-fd-primary" />
            <h2 className="font-semibold">{title}</h2>
            <p className="mt-2 text-sm leading-6 text-fd-muted-foreground">{description}</p>
          </div>
        ))}
      </section>
    </main>
  );
}
