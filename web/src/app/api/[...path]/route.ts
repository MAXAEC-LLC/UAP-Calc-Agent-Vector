import { NextRequest } from "next/server";

const BACKEND =
  process.env.BACKEND_INTERNAL_URL ?? "http://127.0.0.1:8000";

// Allow long-running requests (chat can take ~120s)
export const maxDuration = 300;

async function proxy(req: NextRequest) {
  const path = req.nextUrl.pathname; // e.g. /api/chat
  const search = req.nextUrl.search; // e.g. ?borough=3&block=6767
  const target = `${BACKEND}${path}${search}`;

  const headers = new Headers(req.headers);
  headers.delete("host");

  const init: RequestInit = {
    method: req.method,
    headers,
    // @ts-expect-error — duplex needed for streaming request bodies
    duplex: "half",
  };

  if (req.method !== "GET" && req.method !== "HEAD") {
    init.body = req.body;
  }

  const upstream = await fetch(target, init);

  return new Response(upstream.body, {
    status: upstream.status,
    statusText: upstream.statusText,
    headers: upstream.headers,
  });
}

export const GET = proxy;
export const POST = proxy;
export const PUT = proxy;
export const DELETE = proxy;
export const PATCH = proxy;
