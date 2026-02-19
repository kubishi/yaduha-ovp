import { Container, getContainer } from "@cloudflare/containers";

interface Env {
  YADUHA_CONTAINER: DurableObjectNamespace<YaduhaContainer>;
  OPENAI_API_KEY: string;
  ANTHROPIC_API_KEY: string;
  GEMINI_API_KEY: string;
  ALLOWED_ORIGINS: string;
}

export class YaduhaContainer extends Container<Env> {
  defaultPort = 8000;
  sleepAfter = "5m";

  override get envVars(): Record<string, string> {
    return {
      OPENAI_API_KEY: this.env?.OPENAI_API_KEY ?? "",
      ANTHROPIC_API_KEY: this.env?.ANTHROPIC_API_KEY ?? "",
      GEMINI_API_KEY: this.env?.GEMINI_API_KEY ?? "",
    };
  }
}

export default {
  async fetch(request: Request, env: Env): Promise<Response> {
    if (request.method === "OPTIONS") {
      return handleCors(request, env, new Response(null, { status: 204 }));
    }

    const container = getContainer(env.YADUHA_CONTAINER, "yaduha-api");
    const response = await container.fetch(request);

    return handleCors(request, env, response);
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
