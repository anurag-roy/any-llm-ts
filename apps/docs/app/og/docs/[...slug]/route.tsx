import { ImageResponse } from "next/og";
import { notFound } from "next/navigation";
import { generate as DefaultImage } from "fumadocs-ui/og";
import { appName } from "@/lib/shared";
import { getPageImageUrl, source } from "@/lib/source";

export const revalidate = false;

export async function GET(_request: Request, { params }: RouteContext<"/og/docs/[...slug]">) {
  const { slug } = await params;
  const page = source.getPage(slug.slice(0, -1));
  if (!page) notFound();

  return new ImageResponse(
    <DefaultImage title={page.data.title} description={page.data.description} site={appName} />,
    { height: 630, width: 1200 },
  );
}

export function generateStaticParams() {
  return source.getPages().map((page) => ({ slug: getPageImageUrl(page).segments }));
}
