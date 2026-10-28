import { randomUUID } from "node:crypto";
import { fileURLToPath } from "node:url";

import * as grpc from "@grpc/grpc-js";
import * as protoLoader from "@grpc/proto-loader";
import {
  calculateCost,
  createAssistantMessageEventStream,
  type AssistantMessage,
  type AssistantMessageEventStream,
  type Context,
  type Model,
  type SimpleStreamOptions,
  type Tool,
  type ToolResultMessage,
  type UserMessage,
} from "@earendil-works/pi-ai";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";

type WireRequest = Record<string, unknown>;
type DuplexCall = grpc.ClientDuplexStream<WireRequest, WireResponse>;
type ReasoningEffort =
  | "REASONING_EFFORT_LOW"
  | "REASONING_EFFORT_MEDIUM"
  | "REASONING_EFFORT_XHIGH";

interface WireCompletion {
  reason: string;
  numInputTokens: string | number;
  numOutputTokens: string | number;
}

interface WireResponse {
  reasoning?: { text: string };
  text?: { text: string };
  toolCall?: { id: string; toolId: string; argumentsJson: string };
  completion?: WireCompletion;
}

interface DynamicInferenceClient extends grpc.Client {
  generateMessagesStream(): DuplexCall;
}

interface ActiveTurn {
  emitter: TurnEmitter;
  onResponse?: () => void | Promise<void>;
  responseStarted: boolean;
  resolve(): void;
  reject(error: unknown): void;
}

interface ToolSnapshot {
  id: string;
  description: string;
  inputSchemaJson: string;
}

interface SessionConfiguration {
  enableThinking: boolean;
  reasoningEffort?: ReasoningEffort;
  systemPrompt?: string;
  tools: Map<string, ToolSnapshot>;
}

const PROTO_PATH = fileURLToPath(
  new URL("../../../crates/inference-runtime-proto/proto/inference_runtime.proto", import.meta.url),
);
const packageDefinition = protoLoader.loadSync(PROTO_PATH, {
  defaults: true,
  enums: String,
  longs: String,
  oneofs: true,
});
const proto = grpc.loadPackageDefinition(packageDefinition) as any;
const InferenceClient = proto.khalamari.inference.v1.InferenceRuntime as new (
  target: string,
  credentials: grpc.ChannelCredentials,
) => DynamicInferenceClient;

class ResidentSession {
  private active?: ActiveTurn;
  private readonly call: DuplexCall;
  private events = Promise.resolve();
  private numMessages = 0;
  private responseHeaders: Record<string, string> = {};

  constructor(
    baseUrl: string,
    private readonly configuration: SessionConfiguration,
  ) {
    const url = new URL(baseUrl);
    const credentials = url.protocol === "https:" ? grpc.credentials.createSsl() : grpc.credentials.createInsecure();
    const client = new InferenceClient(url.host, credentials);
    this.call = client.generateMessagesStream();
    this.call.on("metadata", (metadata: grpc.Metadata) => {
      this.responseHeaders = metadataHeaders(metadata);
    });
    this.call.on("data", (response: WireResponse) => this.enqueue(() => this.receive(response)));
    this.call.on("error", (error: grpc.ServiceError) => this.enqueue(() => this.fail(error)));
    this.call.on("end", () => this.enqueue(() => this.fail(new Error("khalamari message stream ended"))));
  }

  matches(configuration: SessionConfiguration): boolean {
    return sameConfiguration(this.configuration, configuration);
  }

  async generate(
    context: Context,
    model: Model<any>,
    options: SimpleStreamOptions | undefined,
    emitter: TurnEmitter,
    firstRequest: boolean,
  ): Promise<void> {
    if (this.active) {
      throw new Error("khalamari session already has an active turn");
    }
    const messages = firstRequest ? context.messages : context.messages.slice(this.numMessages);
    const mappedRequest = mapRequest(
      messages,
      firstRequest ? this.configuration.systemPrompt : undefined,
      model,
      options,
      firstRequest ? fullToolDelta(this.configuration.tools) : { insert: [], remove: [] },
      this.configuration,
    );
    const replacement = await options?.onPayload?.(mappedRequest, model);
    const request = (replacement === undefined ? mappedRequest : replacement) as WireRequest;
    options?.signal?.throwIfAborted();
    await new Promise<void>((resolve, reject) => {
      this.active = {
        emitter,
        onResponse: options?.onResponse
          ? () => options.onResponse?.({ status: 200, headers: this.responseHeaders }, model)
          : undefined,
        responseStarted: false,
        resolve,
        reject,
      };
      this.call.write(request, (error?: Error | null) => {
        if (error) {
          this.enqueue(() => this.fail(error));
        }
      });
    });
    this.numMessages = context.messages.length + 1;
  }

  close(): void {
    this.call.end();
  }

  abort(): void {
    this.call.cancel();
  }

  private enqueue(run: () => void | Promise<void>): void {
    this.events = this.events.then(run).catch((error) => {
      this.fail(error);
      this.call.cancel();
    });
  }

  private async receive(response: WireResponse): Promise<void> {
    const active = this.active;
    if (!active) {
      this.call.cancel();
      return;
    }
    if (!active.responseStarted) {
      active.responseStarted = true;
      await active.onResponse?.();
    }
    if (response.reasoning) {
      active.emitter.text("thinking", response.reasoning.text);
    } else if (response.text) {
      active.emitter.text("text", response.text.text);
    } else if (response.toolCall) {
      active.emitter.toolCall(response.toolCall.id, response.toolCall.toolId, response.toolCall.argumentsJson);
    } else if (response.completion) {
      active.emitter.complete(response.completion);
      this.active = undefined;
      active.resolve();
    } else {
      throw new Error("khalamari returned an empty message event");
    }
  }

  private fail(error: unknown): void {
    const active = this.active;
    this.active = undefined;
    active?.reject(error);
  }
}

class SessionPool {
  private readonly sessions = new Map<string, ResidentSession>();

  async generate(
    model: Model<any>,
    context: Context,
    options: SimpleStreamOptions | undefined,
    emitter: TurnEmitter,
  ): Promise<void> {
    options?.signal?.throwIfAborted();
    const persistent = options?.sessionId !== undefined && options.cacheRetention !== "none";
    const key = persistent ? `${model.baseUrl}\u0000${model.id}\u0000${options.sessionId}` : randomUUID();
    const configuration = sessionConfiguration(context.systemPrompt, context.tools ?? [], model, options);
    let session = this.sessions.get(key);
    if (session && !session.matches(configuration)) {
      session.abort();
      this.sessions.delete(key);
      session = undefined;
    }
    const firstRequest = session === undefined;
    if (!session) {
      session = new ResidentSession(model.baseUrl, configuration);
      this.sessions.set(key, session);
    }
    const abort = () => session?.abort();
    options?.signal?.addEventListener("abort", abort, { once: true });
    try {
      await session.generate(context, model, options, emitter, firstRequest);
    } catch (error) {
      session.abort();
      this.sessions.delete(key);
      throw error;
    } finally {
      options?.signal?.removeEventListener("abort", abort);
      if (!persistent) {
        session.close();
        this.sessions.delete(key);
      }
    }
  }

  shutdownAll(): void {
    for (const session of this.sessions.values()) {
      session.abort();
    }
    this.sessions.clear();
  }
}

class TurnEmitter {
  readonly output: AssistantMessage;
  private open?: { kind: "text" | "thinking"; index: number };
  private sawToolCall = false;

  constructor(
    private readonly stream: AssistantMessageEventStream,
    private readonly model: Model<any>,
  ) {
    this.output = {
      role: "assistant",
      content: [],
      api: model.api,
      provider: model.provider,
      model: model.id,
      usage: {
        input: 0,
        output: 0,
        cacheRead: 0,
        cacheWrite: 0,
        totalTokens: 0,
        cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
      },
      stopReason: "pending",
      timestamp: Date.now(),
    };
  }

  start(): void {
    this.stream.push({ type: "start", partial: this.output });
  }

  text(kind: "text" | "thinking", delta: string): void {
    if (!delta) return;
    if (this.open?.kind !== kind) {
      this.closeOpen();
      const index = this.output.content.length;
      if (kind === "text") {
        this.output.content.push({ type: "text", text: "" });
        this.stream.push({ type: "text_start", contentIndex: index, partial: this.output });
      } else {
        this.output.content.push({ type: "thinking", thinking: "" });
        this.stream.push({ type: "thinking_start", contentIndex: index, partial: this.output });
      }
      this.open = { kind, index };
    }
    const index = this.open.index;
    const block = this.output.content[index];
    if (kind === "text" && block.type === "text") {
      block.text += delta;
      this.stream.push({ type: "text_delta", contentIndex: index, delta, partial: this.output });
    } else if (kind === "thinking" && block.type === "thinking") {
      block.thinking += delta;
      this.stream.push({ type: "thinking_delta", contentIndex: index, delta, partial: this.output });
    } else {
      throw new Error("khalamari content stream changed block type");
    }
  }

  toolCall(id: string, name: string, argumentsJson: string): void {
    this.sawToolCall = true;
    this.closeOpen();
    const argumentsValue = JSON.parse(argumentsJson) as Record<string, unknown>;
    const index = this.output.content.length;
    const toolCall = { type: "toolCall" as const, id, name, arguments: argumentsValue };
    this.output.content.push(toolCall);
    this.stream.push({ type: "toolcall_start", contentIndex: index, partial: this.output });
    this.stream.push({ type: "toolcall_delta", contentIndex: index, delta: argumentsJson, partial: this.output });
    this.stream.push({ type: "toolcall_end", contentIndex: index, toolCall, partial: this.output });
  }

  complete(completion: WireCompletion): void {
    this.closeOpen();
    switch (completion.reason) {
      case "COMPLETION_REASON_STOP_SEQUENCE":
        this.output.stopReason = this.sawToolCall ? "toolUse" : "stop";
        break;
      case "COMPLETION_REASON_LENGTH_LIMIT":
        this.output.stopReason = this.sawToolCall ? "toolUse" : "length";
        break;
      case "COMPLETION_REASON_CONTEXT_LIMIT":
        throw new Error("context_length_exceeded: khalamari reached the model context limit");
      default:
        throw new Error(`khalamari returned unexpected completion reason: ${completion.reason}`);
    }
    this.output.usage.input = Number(completion.numInputTokens);
    this.output.usage.output = Number(completion.numOutputTokens);
    this.output.usage.totalTokens = this.output.usage.input + this.output.usage.output;
    calculateCost(this.model, this.output.usage);
  }

  done(): void {
    const reason = this.output.stopReason;
    if (reason !== "stop" && reason !== "length" && reason !== "toolUse") {
      throw new Error("khalamari stream ended without a completion event");
    }
    this.stream.push({ type: "done", reason, message: this.output });
    this.stream.end();
  }

  error(error: unknown, aborted: boolean): void {
    this.closeOpen();
    this.output.stopReason = aborted ? "aborted" : "error";
    this.output.errorMessage = error instanceof Error ? error.message : String(error);
    this.stream.push({ type: "error", reason: this.output.stopReason, error: this.output });
    this.stream.end();
  }

  private closeOpen(): void {
    const open = this.open;
    if (!open) return;
    const block = this.output.content[open.index];
    if (open.kind === "text" && block.type === "text") {
      this.stream.push({ type: "text_end", contentIndex: open.index, content: block.text, partial: this.output });
    } else if (open.kind === "thinking" && block.type === "thinking") {
      this.stream.push({
        type: "thinking_end",
        contentIndex: open.index,
        content: block.thinking,
        partial: this.output,
      });
    }
    this.open = undefined;
  }
}

function streamMessages(
  sessions: SessionPool,
  model: Model<any>,
  context: Context,
  options?: SimpleStreamOptions,
): AssistantMessageEventStream {
  const stream = createAssistantMessageEventStream();
  const emitter = new TurnEmitter(stream, model);
  emitter.start();
  void sessions
    .generate(model, context, options, emitter)
    .then(() => emitter.done())
    .catch((error) => emitter.error(error, options?.signal?.aborted === true));
  return stream;
}

function mapRequest(
  messages: Context["messages"],
  systemPrompt: string | undefined,
  model: Model<any>,
  options: SimpleStreamOptions | undefined,
  tools: { insert: ToolSnapshot[]; remove: string[] },
  configuration: SessionConfiguration,
): WireRequest {
  const sampling = { ...(model.samplingParams ?? {}), ...(options?.samplingParams ?? {}) };
  return {
    messages: mapMessages(messages, systemPrompt),
    tools: { insert: tools.insert, remove: tools.remove },
    generation: {
      maxSampledTokens: options?.maxTokens ?? model.maxTokens,
      temperature: options?.temperature ?? numberOption(sampling.temperature),
      topK: numberOption(sampling.top_k),
      topP: numberOption(sampling.top_p),
      seed: numberOption(sampling.seed),
      stopSequences: [],
    },
    enableThinking: configuration.enableThinking,
    reasoningEffort: configuration.reasoningEffort,
  };
}

function mapMessages(input: Context["messages"], systemPrompt: string | undefined): WireRequest[] {
  const messages: WireRequest[] = [];
  if (systemPrompt) {
    messages.push({ system: { text: systemPrompt } });
  }
  for (const message of input) {
    if (message.role === "user") {
      messages.push({ user: { text: textContent(message.content, "user message") } });
    } else if (message.role === "assistant") {
      const content: WireRequest[] = [];
      const toolCalls: WireRequest[] = [];
      for (const block of message.content) {
        if (block.type === "text") {
          content.push({ text: block.text });
        } else if (block.type === "thinking") {
          content.push({ reasoning: block.thinking });
        } else {
          toolCalls.push({ id: block.id, toolId: block.name, argumentsJson: JSON.stringify(block.arguments) });
        }
      }
      messages.push({ assistant: { content, toolCalls } });
    } else {
      messages.push({
        toolResult: {
          toolCallId: message.toolCallId,
          toolId: message.toolName,
          text: [textContent(message.content, "tool result")],
          isError: message.isError,
        },
      });
    }
  }
  return messages;
}

function textContent(content: UserMessage["content"] | ToolResultMessage["content"], owner: string): string {
  if (typeof content === "string") return content;
  let text = "";
  for (const block of content) {
    if (block.type !== "text") {
      throw new Error(`khalamari ${owner} does not support image content`);
    }
    text += block.text;
  }
  return text;
}

function snapshotTools(tools: Tool[]): Map<string, ToolSnapshot> {
  const snapshot = new Map<string, ToolSnapshot>();
  for (const tool of tools) {
    snapshot.set(tool.name, {
      id: tool.name,
      description: tool.description,
      inputSchemaJson: JSON.stringify(tool.parameters),
    });
  }
  return snapshot;
}

function fullToolDelta(tools: Map<string, ToolSnapshot>): { insert: ToolSnapshot[]; remove: string[] } {
  return { insert: [...tools.values()], remove: [] };
}

function sessionConfiguration(
  systemPrompt: string | undefined,
  tools: Tool[],
  model: Model<any>,
  options: SimpleStreamOptions | undefined,
): SessionConfiguration {
  const enableThinking = model.reasoning && options?.reasoning !== undefined;
  return {
    enableThinking,
    reasoningEffort: enableThinking ? mapReasoning(options?.reasoning) : undefined,
    systemPrompt: systemPrompt || undefined,
    tools: snapshotTools(tools),
  };
}

function sameConfiguration(left: SessionConfiguration, right: SessionConfiguration): boolean {
  if (
    left.enableThinking !== right.enableThinking ||
    left.reasoningEffort !== right.reasoningEffort ||
    left.systemPrompt !== right.systemPrompt
  ) {
    return false;
  }
  if (left.tools.size !== right.tools.size) {
    return false;
  }
  for (const [id, tool] of left.tools) {
    const other = right.tools.get(id);
    if (!other || tool.description !== other.description || tool.inputSchemaJson !== other.inputSchemaJson) {
      return false;
    }
  }
  return true;
}

function metadataHeaders(metadata: grpc.Metadata): Record<string, string> {
  return Object.fromEntries(
    Object.entries(metadata.getMap()).map(([name, value]) => [
      name,
      Buffer.isBuffer(value) ? value.toString("base64") : value,
    ]),
  );
}

function numberOption(value: unknown): number | undefined {
  return typeof value === "number" && Number.isFinite(value) ? value : undefined;
}

function mapReasoning(reasoning: SimpleStreamOptions["reasoning"]): ReasoningEffort | undefined {
  switch (reasoning) {
    case "minimal":
    case "low":
      return "REASONING_EFFORT_LOW";
    case "medium":
      return "REASONING_EFFORT_MEDIUM";
    case "high":
    case "xhigh":
    case "max":
      return "REASONING_EFFORT_XHIGH";
    default:
      return undefined;
  }
}

export default function register(pi: ExtensionAPI): void {
  const sessions = new SessionPool();
  pi.registerProvider("khalamari", {
    name: "khalamari",
    apiKey: "unused",
    api: "khalamari-messages" as any,
    streamSimple: (model, context, options) => streamMessages(sessions, model, context, options),
  });
  pi.on("session_shutdown", async () => sessions.shutdownAll());
}
