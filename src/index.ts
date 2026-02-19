import { Container, getContainer } from "@cloudflare/containers";
import { env } from "cloudflare:workers";

export class YaduhaContainer extends Container<Env> {
  defaultPort = 8000;
  sleepAfter = "5m";
  envVars = {
    OPENAI_API_KEY: env.OPENAI_API_KEY ?? "",
    ANTHROPIC_API_KEY: env.ANTHROPIC_API_KEY ?? "",
    GEMINI_API_KEY: env.GEMINI_API_KEY ?? "",
  };

  override onStart() {
    console.log("Yaduha container started");
  }

  override onStop() {
    console.log("Yaduha container stopped");
  }

  override onError(error: unknown) {
    console.error("Yaduha container error:", error);
  }
}

export default {
  async fetch(request: Request, env: Env): Promise<Response> {
    if (request.method === "OPTIONS") {
      return handleCors(request, env, new Response(null, { status: 204 }));
    }

    try {
      const container = getContainer(env.YADUHA_CONTAINER, "yaduha-api");
      const response = await container.fetch(request);
      return handleCors(request, env, response);
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : String(e);
      return handleCors(
        request,
        env,
        new Response(JSON.stringify({ error: msg }), {
          status: 502,
          headers: { "Content-Type": "application/json" },
        })
      );
    }
  },
};

function handleCors(
  request: Request,
  env: Env,
  response: Response
): Response {
  const origin = request.headers.get("Origin") ?? "";
  const allowedOrigins = (
    env.ALLOWED_ORIGINS ?? "https://sentences.kubishi.com"
  )
    .split(",")
    .map((o) => o.trim());

  const headers = new Headers(response.headers);

  if (allowedOrigins.includes(origin) || allowedOrigins.includes("*")) {
    headers.set("Access-Control-Allow-Origin", origin);
    headers.set(
      "Access-Control-Allow-Methods",
      "GET, POST, PUT, DELETE, OPTIONS"
    );
    headers.set(
      "Access-Control-Allow-Headers",
      "Content-Type, x-api-key, x-openai-key, x-anthropic-key, x-gemini-key"
    );
    headers.set("Access-Control-Max-Age", "86400");
  }

  return new Response(response.body, {
    status: response.status,
    statusText: response.statusText,
    headers,
  });
}
