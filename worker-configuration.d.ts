interface Env {
  YADUHA_CONTAINER: DurableObjectNamespace;
  OPENAI_API_KEY: string;
  ANTHROPIC_API_KEY: string;
  GEMINI_API_KEY: string;
  ALLOWED_ORIGINS: string;
}

declare module "cloudflare:workers" {
  const env: Env;
}
