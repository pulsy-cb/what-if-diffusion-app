export type ToolCall = { id: string; name: string; args: Record<string, unknown> };

export type ImagePart = { mime: string; data: string }; // base64, no data: prefix

export type ChatMessage =
  | { role: 'system'; content: string }
  | { role: 'user'; content: string; images?: ImagePart[] }
  | { role: 'assistant'; content: string; toolCalls?: ToolCall[] }
  | { role: 'tool'; content: string; toolCallId: string; name: string };

export type ToolSchema = { type: 'function'; function: { name: string; description: string; parameters: unknown } };

export type ChatRequest = {
  messages: ChatMessage[];
  tools: ToolSchema[];
  signal?: AbortSignal;
  onText?: (delta: string) => void;
};

export type ChatResult = {
  text: string;
  toolCalls: ToolCall[];
  usage?: unknown;
  finishReason?: string;
  truncatedToolCall?: string;
};

export type Provider = {
  id: string;
  label: string;
  model: string;
  chat(req: ChatRequest): Promise<ChatResult>;
};

export type ProviderConfig = {
  maxTokens?: number;
  provider: 'mock' | 'openai' | 'anthropic' | 'codex' | 'agy' | string;
  baseURL: string;
  apiKey: string;
  model: string;
};

/** Parse a text/event-stream body into `data:` payload strings, aborting stalled streams after idleTimeoutMs. */
export async function* sseData(
  body: unknown,
  opts?: { idleTimeoutMs?: number; signal?: AbortSignal },
): AsyncGenerator<string> {
  const decoder = new TextDecoder();
  let buf = '';

  const drainBlocks = function* (): Generator<string> {
    if (buf.includes('\r')) buf = buf.replace(/\r\n/g, '\n').replace(/\r/g, '\n');
    let idx: number;
    while ((idx = buf.indexOf('\n\n')) >= 0) {
      const block = buf.slice(0, idx);
      buf = buf.slice(idx + 2);
      const data = block
        .split('\n')
        .filter((l) => l.startsWith('data:'))
        .map((l) => l.slice(5).trim())
        .join('\n');
      if (data) yield data;
    }
  };

  const idleMs = opts?.idleTimeoutMs ?? 180_000;
  const readable = body as {
    getReader?: () => {
      read: () => Promise<{ done: boolean; value?: Uint8Array }>;
      cancel: () => Promise<void>;
      releaseLock: () => void;
    };
  } | null;

  if (readable && typeof readable.getReader === 'function') {
    const reader = readable.getReader();
    const readWithIdle = (): Promise<{ done: boolean; value?: Uint8Array }> =>
      new Promise((resolve, reject) => {
        if (opts?.signal?.aborted) {
          reader.cancel().catch(() => {});
          return reject(new Error('aborted'));
        }
        let settled = false;
        const onAbort = () => {
          if (settled) return;
          settled = true;
          clearTimeout(timer);
          reader.cancel().catch(() => {});
          reject(new Error('aborted'));
        };
        const timer = setTimeout(() => {
          if (settled) return;
          settled = true;
          opts?.signal?.removeEventListener('abort', onAbort);
          reader.cancel().catch(() => {});
          reject(new Error(`SSE stream idle timeout after ${Math.round(idleMs / 1000)}s`));
        }, idleMs);
        opts?.signal?.addEventListener('abort', onAbort, { once: true });
        reader.read().then(
          (res) => {
            if (settled) return;
            settled = true;
            clearTimeout(timer);
            opts?.signal?.removeEventListener('abort', onAbort);
            resolve(res);
          },
          (err) => {
            if (settled) return;
            settled = true;
            clearTimeout(timer);
            opts?.signal?.removeEventListener('abort', onAbort);
            reject(err);
          },
        );
      });

    try {
      while (true) {
        const { done, value } = await readWithIdle();
        if (done) break;
        if (value) {
          buf += decoder.decode(value, { stream: true });
          yield* drainBlocks();
        }
      }
    } finally {
      try {
        reader.releaseLock();
      } catch {
        /* ignore */
      }
    }
  } else {
    const stream = body as AsyncIterable<Uint8Array>;
    for await (const chunk of stream) {
      buf += decoder.decode(chunk, { stream: true });
      yield* drainBlocks();
    }
  }

  buf += decoder.decode();
  if (buf.includes('\r')) buf = buf.replace(/\r\n/g, '\n').replace(/\r/g, '\n');
  if (buf.trim()) {
    const data = buf
      .split('\n')
      .filter((l) => l.startsWith('data:'))
      .map((l) => l.slice(5).trim())
      .join('\n');
    if (data) yield data;
  }
}

export function safeJson(s: string): any {
  try {
    return JSON.parse(s);
  } catch {
    return null;
  }
}
