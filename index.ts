import {
	calculateCost, createAssistantMessageEventStream, getModels,
	type AssistantMessage, type AssistantMessageEventStream,
	type Context, type Model, type SimpleStreamOptions, type Tool,
} from "@earendil-works/pi-ai";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import {
	createSdkMcpServer, query,
	type EffortLevel, type SDKMessage, type SDKUserMessage,
} from "@anthropic-ai/claude-agent-sdk";
import type { Base64ImageSource, ContentBlockParam, MessageParam } from "@anthropic-ai/sdk/resources";
import { z } from "zod";
import { pascalCase } from "change-case";
import { createSession, getSessionPath } from "cc-session-io";
import { existsSync, readFileSync } from "fs";
import { homedir } from "os";
import { dirname, join, relative, resolve } from "path";

// --- Constants ---

const PROVIDER_ID = "claude-sdk";
const MCP_SERVER_NAME = "custom-tools";
const MCP_TOOL_PREFIX = `mcp__${MCP_SERVER_NAME}__`;

const DISALLOWED_BUILTIN_TOOLS = [
	"Read", "Write", "Edit", "Glob", "Grep", "Bash", "Agent",
	"NotebookEdit", "EnterWorktree", "ExitWorktree",
	"CronCreate", "CronDelete", "CronList", "TeamCreate", "TeamDelete",
	"WebFetch", "WebSearch", "TodoRead", "TodoWrite",
	"EnterPlanMode", "ExitPlanMode", "RemoteTrigger", "SendMessage",
	"Skill", "TaskOutput", "TaskStop", "ToolSearch",
	"AskUserQuestion", "TaskCreate", "TaskGet", "TaskList", "TaskUpdate",
];

const SDK_TO_PI: Record<string, string> = { read: "read", write: "write", edit: "edit", bash: "bash", grep: "grep", glob: "find" };
const PI_TO_SDK: Record<string, string> = { read: "Read", write: "Write", edit: "Edit", bash: "Bash", grep: "Grep", find: "Glob", glob: "Glob" };

const LATEST_MODEL_IDS = new Set(["claude-opus-4-7", "claude-opus-4-6", "claude-sonnet-4-6", "claude-haiku-4-5"]);
const DEFAULT_1M_OPUS_MODEL_IDS = new Set(["claude-opus-4-7", "claude-opus-4-6"]);

function uses1mContextByDefault(modelId: string): boolean {
	const base = modelId.replace(/\[1m\]$/i, "");
	return DEFAULT_1M_OPUS_MODEL_IDS.has(base);
}

function toClaudeSdkModelId(modelId: string): string {
	if (!uses1mContextByDefault(modelId) || /\[1m\]$/i.test(modelId)) return modelId;

	// Claude Code treats the [1m] suffix as the switch for the extended-context
	// variant. Use the opus alias for the newest Opus so the bundled SDK/CLI can
	// resolve it to whatever the user's Claude Code currently considers latest.
	if (modelId === "claude-opus-4-7") return "opus[1m]";
	return `${modelId}[1m]`;
}

const MODELS = getModels("anthropic")
	.filter((m) => LATEST_MODEL_IDS.has(m.id))
	.map(({ id, name, reasoning, thinkingLevelMap, input, cost, contextWindow, maxTokens }) => ({
		id,
		name: uses1mContextByDefault(id) && !/1m/i.test(name) ? `${name} (1M context)` : name,
		reasoning,
		thinkingLevelMap,
		input,
		cost,
		contextWindow: uses1mContextByDefault(id) ? 1_000_000 : contextWindow,
		maxTokens,
	}));

const EFFORT: Record<string, EffortLevel> = { minimal: "low", low: "low", medium: "medium", high: "high", xhigh: "max" };

// Skill alias paths — CC sees ~/.claude/skills, pi stores at ~/.pi/agent/skills
const SKILLS_ALIAS_GLOBAL = "~/.claude/skills";
const SKILLS_ALIAS_PROJECT = ".claude/skills";
const GLOBAL_SKILLS_ROOT = join(homedir(), ".pi", "agent", "skills");
const PROJECT_SKILLS_ROOT = join(process.cwd(), ".pi", "skills");
const GLOBAL_AGENTS_PATH = join(homedir(), ".pi", "agent", "AGENTS.md");

// --- Helpers ---

function textOf(content: string | Array<{ type: string; text?: string }>): string {
	if (typeof content === "string") return content;
	return (content ?? []).filter((b) => b.type === "text" && b.text).map((b) => b.text!).join("\n");
}

type McpContent = Array<{ type: "text"; text: string } | { type: "image"; data: string; mimeType: string }>;

function toMcpContent(content: string | Array<{ type: string; text?: string; data?: string; mimeType?: string }>): McpContent {
	if (typeof content === "string") return [{ type: "text", text: content || "" }];
	if (!Array.isArray(content)) return [{ type: "text", text: "" }];
	const r: McpContent = [];
	for (const b of content) {
		if (b.type === "text" && b.text) r.push({ type: "text", text: b.text });
		else if (b.type === "image" && b.data && b.mimeType) r.push({ type: "image", data: b.data, mimeType: b.mimeType });
	}
	return r.length ? r : [{ type: "text", text: "" }];
}

function extractToolResults(ctx: Context): McpResult[] {
	const results: McpResult[] = [];
	for (let i = ctx.messages.length - 1; i >= 0; i--) {
		const m = ctx.messages[i];
		if (m.role === "toolResult") results.unshift({ content: toMcpContent(m.content), isError: m.isError });
		else if (m.role === "user" || m.role === "assistant") break;
	}
	return results;
}

function userPrompt(msgs: Context["messages"]): string | null {
	const last = msgs[msgs.length - 1];
	return last?.role === "user" ? (typeof last.content === "string" ? last.content : textOf(last.content)) || "" : null;
}

function userPromptBlocks(msgs: Context["messages"]): ContentBlockParam[] | null {
	const last = msgs[msgs.length - 1];
	if (!last || last.role !== "user" || typeof last.content === "string" || !Array.isArray(last.content)) return null;
	let hasImage = false;
	const blocks: ContentBlockParam[] = [];
	for (const b of last.content) {
		if (b.type === "text" && b.text) blocks.push({ type: "text", text: b.text });
		else if (b.type === "image" && (b as any).data && (b as any).mimeType) {
			hasImage = true;
			blocks.push({ type: "image", source: { type: "base64", media_type: b.mimeType as Base64ImageSource["media_type"], data: b.data } });
		}
	}
	return hasImage ? blocks : null;
}

async function* wrapPromptStream(blocks: ContentBlockParam[]): AsyncIterable<SDKUserMessage> {
	yield { type: "user", message: { role: "user", content: blocks } as MessageParam, parent_tool_use_id: null };
}

// --- System prompt: AGENTS.md + Skills forwarding ---
//
// CC needs to see the user's project instructions (AGENTS.md) and pi's available
// skills. Without this, CC runs with only its default system prompt — no project
// context, no skill awareness.

function resolveAgentsMdPath(): string | undefined {
	let current = resolve(process.cwd());
	while (true) {
		const candidate = join(current, "AGENTS.md");
		if (existsSync(candidate)) return candidate;
		const parent = dirname(current);
		if (parent === current) break;
		current = parent;
	}
	return existsSync(GLOBAL_AGENTS_PATH) ? GLOBAL_AGENTS_PATH : undefined;
}

function extractAgentsAppend(): string | undefined {
	const agentsPath = resolveAgentsMdPath();
	if (!agentsPath) return undefined;
	try {
		const content = readFileSync(agentsPath, "utf-8").trim();
		if (!content) return undefined;
		// Sanitize pi references → claude equivalents so CC understands them
		let sanitized = content;
		sanitized = sanitized.replace(/~\/\.pi\b/gi, "~/.claude");
		sanitized = sanitized.replace(/(^|[\s'"`])\.pi\//g, "$1.claude/");
		sanitized = sanitized.replace(/\b\.pi\b/gi, ".claude");
		sanitized = sanitized.replace(/\bpi\b/gi, "environment");
		return `# CLAUDE.md\n\n${sanitized}`;
	} catch { return undefined; }
}

function rewriteSkillsLocations(skillsBlock: string): string {
	return skillsBlock.replace(/<location>([^<]+)<\/location>/g, (_match, location: string) => {
		let rewritten = location;
		if (location.startsWith(GLOBAL_SKILLS_ROOT)) {
			const relPath = relative(GLOBAL_SKILLS_ROOT, location).replace(/^\.+/, "");
			rewritten = `${SKILLS_ALIAS_GLOBAL}/${relPath}`.replace(/\/\/+/g, "/");
		} else if (location.startsWith(PROJECT_SKILLS_ROOT)) {
			const relPath = relative(PROJECT_SKILLS_ROOT, location).replace(/^\.+/, "");
			rewritten = `${SKILLS_ALIAS_PROJECT}/${relPath}`.replace(/\/\/+/g, "/");
		}
		return `<location>${rewritten}</location>`;
	});
}

function extractSkillsBlock(systemPrompt?: string): string | undefined {
	if (!systemPrompt) return undefined;
	const startMarker = "The following skills provide specialized instructions for specific tasks.";
	const endMarker = "</available_skills>";
	const start = systemPrompt.indexOf(startMarker);
	if (start === -1) return undefined;
	const end = systemPrompt.indexOf(endMarker, start);
	if (end === -1) return undefined;
	return rewriteSkillsLocations(systemPrompt.slice(start, end + endMarker.length).trim());
}

// --- Tool name/arg mapping ---

function toSdkName(name: string, custom?: Map<string, string>): string {
	if (custom) { const m = custom.get(name) ?? custom.get(name.toLowerCase()); if (m) return m; }
	return PI_TO_SDK[name.toLowerCase()] ?? pascalCase(name);
}

function toPiName(name: string, custom?: Map<string, string>): string {
	const lo = name.toLowerCase();
	if (SDK_TO_PI[lo]) return SDK_TO_PI[lo];
	if (custom?.has(name) || custom?.has(lo)) return custom!.get(name) ?? custom!.get(lo)!;
	if (lo.startsWith(MCP_TOOL_PREFIX)) return name.slice(MCP_TOOL_PREFIX.length);
	return name;
}

// Rewrite skill alias paths: CC thinks skills live at ~/.claude/skills but
// pi stores them at ~/.pi/agent/skills. When CC reads a skill file, the path
// needs rewriting so pi can find it.
function rewriteSkillAliasPath(pathValue: unknown): unknown {
	if (typeof pathValue !== "string") return pathValue;
	if (pathValue.startsWith(SKILLS_ALIAS_GLOBAL)) return pathValue.replace(SKILLS_ALIAS_GLOBAL, "~/.pi/agent/skills");
	if (pathValue.startsWith(`./${SKILLS_ALIAS_PROJECT}`)) return pathValue.replace(`./${SKILLS_ALIAS_PROJECT}`, PROJECT_SKILLS_ROOT);
	if (pathValue.startsWith(SKILLS_ALIAS_PROJECT)) return pathValue.replace(SKILLS_ALIAS_PROJECT, PROJECT_SKILLS_ROOT);
	const projectAliasAbs = join(process.cwd(), SKILLS_ALIAS_PROJECT);
	if (pathValue.startsWith(projectAliasAbs)) return pathValue.replace(projectAliasAbs, PROJECT_SKILLS_ROOT);
	return pathValue;
}

function mapArgs(name: string, args?: Record<string, unknown>, rewriteSkills = false): Record<string, unknown> {
	const i = args ?? {};
	const rp = (v: unknown) => rewriteSkills ? rewriteSkillAliasPath(v) : v;
	switch (name.toLowerCase()) {
		case "read": return { path: rp(i.file_path ?? i.path), offset: i.offset, limit: i.limit };
		case "write": return { path: rp(i.file_path ?? i.path), content: i.content };
		case "edit": {
			const path = rp(i.file_path ?? i.path);
			// Pass through pi-native params directly
			if (Array.isArray(i.multi)) return { path, multi: i.multi };
			if (typeof i.patch === "string") return { path, patch: i.patch };
			// Map CC-native (old_string/new_string) or pi-native (oldText/newText) → top-level
			const o = i.old_string ?? i.oldText ?? i.old_text;
			const n = i.new_string ?? i.newText ?? i.new_text;
			if (typeof o === "string" && typeof n === "string") return { path, oldText: o, newText: n };
			return { path, ...i };
		}
		case "bash": return { command: i.command, timeout: i.timeout ?? 120 };
		case "grep": return { pattern: i.pattern, path: rp(i.path), glob: i.glob, limit: i.head_limit ?? i.limit };
		case "find": return { pattern: i.pattern, path: rp(i.path) };
		default: return i;
	}
}

// --- Session persistence ---
//
// Converts pi's message history into an Anthropic-format CC session file so
// query() can resume from it. This gives CC full conversation context and
// enables Anthropic's automatic prompt caching (repeated prefixes are cached,
// subsequent turns pay only cache_read instead of full input tokens).

interface SessionState { sessionId: string; cursor: number; cwd: string }
let session: SessionState | null = null;

function importMessages(
	ccSession: ReturnType<typeof createSession>,
	msgs: Context["messages"],
	toSdkMap?: Map<string, string>,
): void {
	const anthropic: Array<{ role: string; content: unknown }> = [];
	const idCache = new Map<string, string>();
	const cleanId = (id: string) => {
		if (idCache.has(id)) return idCache.get(id)!;
		const clean = id.replace(/[^a-zA-Z0-9_-]/g, "_");
		idCache.set(id, clean);
		return clean;
	};

	for (const msg of msgs) {
		if (msg.role === "user") {
			if (typeof msg.content === "string") {
				anthropic.push({ role: "user", content: msg.content || "[empty]" });
			} else if (Array.isArray(msg.content)) {
				const parts: unknown[] = [];
				for (const b of msg.content) {
					if (b.type === "text" && b.text) parts.push({ type: "text", text: b.text });
					else if (b.type === "image" && b.data && b.mimeType)
						parts.push({ type: "image", source: { type: "base64", media_type: b.mimeType, data: b.data } });
				}
				anthropic.push({ role: "user", content: parts.length ? parts : "[image]" });
			} else {
				anthropic.push({ role: "user", content: "[empty]" });
			}
		} else if (msg.role === "assistant") {
			const blocks: unknown[] = [];
			for (const b of Array.isArray(msg.content) ? msg.content : []) {
				if (b.type === "text" && b.text) {
					blocks.push({ type: "text", text: b.text });
				} else if (b.type === "thinking") {
					const sig = (b as any).thinkingSignature;
					const isOurs = (msg as any).provider === PROVIDER_ID || (msg as any).api === "anthropic";
					if (isOurs && sig) blocks.push({ type: "thinking", thinking: b.thinking ?? "", signature: sig });
				} else if (b.type === "toolCall") {
					blocks.push({ type: "tool_use", id: cleanId(b.id), name: toSdkName(b.name, toSdkMap), input: b.arguments ?? {} });
				}
			}
			if (blocks.length) anthropic.push({ role: "assistant", content: blocks });
		} else if (msg.role === "toolResult") {
			const text = typeof msg.content === "string" ? msg.content : textOf(msg.content);
			anthropic.push({ role: "user", content: [{ type: "tool_result", tool_use_id: cleanId(msg.toolCallId), content: text || "", is_error: msg.isError }] });
		}
	}

	if (anthropic.length) ccSession.importMessages(anthropic as any);
}

function syncSession(
	msgs: Context["messages"], cwd: string, toSdkMap?: Map<string, string>, modelId?: string,
): string | null {
	const prior = msgs.slice(0, -1);

	// No prior history → nothing to resume.
	if (prior.length === 0) {
		session = null;
		return null;
	}

	// Sessions are project-path scoped in Claude Code. If cwd changed, any old
	// session ID may be invisible from the new cwd. Rebuild under the current cwd.
	if (!session || session.cwd !== cwd) {
		const s = createSession({ projectPath: cwd, ...(modelId ? { model: modelId } : {}) });
		importMessages(s, prior, toSdkMap);
		s.save();
		session = { sessionId: s.sessionId, cursor: prior.length, cwd };
		return s.sessionId;
	}

	// If the backing JSONL file is gone, rebuild instead of trusting a stale ID.
	if (!existsSync(getSessionPath(session.sessionId, cwd))) {
		const s = createSession({ projectPath: cwd, ...(modelId ? { model: modelId } : {}) });
		importMessages(s, prior, toSdkMap);
		s.save();
		session = { sessionId: s.sessionId, cursor: prior.length, cwd };
		return s.sessionId;
	}

	if (prior.length <= session.cursor) return session.sessionId;

	const s = createSession({ projectPath: cwd, ...(modelId ? { model: modelId } : {}) });
	importMessages(s, prior, toSdkMap);
	s.save();
	session = { sessionId: s.sessionId, cursor: prior.length, cwd };
	return s.sessionId;
}

// --- MCP bridge state ---

interface McpResult { [key: string]: unknown; content: McpContent; isError?: boolean }
interface PendingToolCall { resolve: (r: McpResult) => void }

let activeQuery: ReturnType<typeof query> | null = null;
let currentStream: ReturnType<typeof createAssistantMessageEventStream> | null = null;
let pendingCalls: PendingToolCall[] = [];
let pendingResults: McpResult[] = [];

// Reentrant state stack: when a subagent starts a fresh query while the parent
// is suspended on a tool call, we save the parent's state and restore it after.
// Without this, the subagent's query overwrites activeQuery/pendingCalls and the
// parent's tool result delivery breaks silently.
interface SavedQueryState {
	activeQuery: typeof activeQuery;
	currentStream: typeof currentStream;
	pendingCalls: PendingToolCall[];
	pendingResults: McpResult[];
}
const queryStateStack: SavedQueryState[] = [];

// --- Turn state ---

let out: AssistantMessage | null = null;
let blocks: any[] = [];
let started = false;
let sawStream = false;
let sawTool = false;

// --- MCP server ---

function resolveMcpTools(ctx: Context) {
	const tools: Tool[] = [];
	const toSdk = new Map<string, string>();
	const toPi = new Map<string, string>();
	for (const t of ctx.tools ?? []) {
		const sdk = `${MCP_TOOL_PREFIX}${t.name}`;
		tools.push(t);
		toSdk.set(t.name, sdk); toSdk.set(t.name.toLowerCase(), sdk);
		toPi.set(sdk, t.name); toPi.set(sdk.toLowerCase(), t.name);
	}
	return { tools, toSdk, toPi };
}

// --- TypeBox → Zod ---

function toZodProp(p: Record<string, unknown>): z.ZodTypeAny {
	let b: z.ZodTypeAny;
	if (Array.isArray(p.enum)) b = z.enum(p.enum as [string, ...string[]]);
	else switch (p.type) {
		case "string": b = z.string(); break;
		case "number": case "integer": b = z.number(); break;
		case "boolean": b = z.boolean(); break;
		case "array": b = p.items ? z.array(toZodProp(p.items as Record<string, unknown>)) : z.array(z.unknown()); break;
		case "object": b = z.record(z.string(), z.unknown()); break;
		default: b = z.unknown();
	}
	return typeof p.description === "string" ? b.describe(p.description) : b;
}

function toZodShape(schema: unknown): Record<string, z.ZodTypeAny> {
	const s = schema as Record<string, unknown>;
	if (!s || s.type !== "object" || !s.properties) return {};
	const props = s.properties as Record<string, Record<string, unknown>>;
	const req = new Set(Array.isArray(s.required) ? s.required as string[] : []);
	const shape: Record<string, z.ZodTypeAny> = {};
	for (const [k, p] of Object.entries(props)) shape[k] = req.has(k) ? toZodProp(p) : toZodProp(p).optional();
	return shape;
}

function buildMcp(tools: Tool[]) {
	if (!tools.length) return undefined;
	const server = createSdkMcpServer({
		name: MCP_SERVER_NAME, version: "1.0.0",
		tools: tools.map((t) => ({
			name: t.name, description: t.description, inputSchema: toZodShape(t.parameters),
			handler: async () => pendingResults.length
				? pendingResults.shift()!
				: new Promise<McpResult>((resolve) => { pendingCalls.push({ resolve }); }),
		})),
	});
	return { [MCP_SERVER_NAME]: server };
}

// --- Usage ---

function addUsage(o: AssistantMessage, u: Record<string, number | undefined>, m: Model<any>) {
	if (u.input_tokens != null) o.usage.input = u.input_tokens;
	if (u.output_tokens != null) o.usage.output = u.output_tokens;
	if (u.cache_read_input_tokens != null) o.usage.cacheRead = u.cache_read_input_tokens;
	if (u.cache_creation_input_tokens != null) o.usage.cacheWrite = u.cache_creation_input_tokens;
	o.usage.totalTokens = o.usage.input + o.usage.output + o.usage.cacheRead + o.usage.cacheWrite;
	calculateCost(m, o.usage);
}

// --- Stream processing ---

function tryJson(s: string, fallback: Record<string, unknown>): Record<string, unknown> {
	try { return JSON.parse(s); } catch { return fallback; }
}

function reset(model: Model<any>) {
	out = {
		role: "assistant", content: [], api: model.api, provider: model.provider, model: model.id,
		usage: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, totalTokens: 0,
			cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 } },
		stopReason: "stop", timestamp: Date.now(),
	};
	blocks = out.content as any[];
	started = sawStream = sawTool = false;
}

function ensureStarted() {
	if (!started && currentStream && out) { currentStream.push({ type: "start", partial: out }); started = true; }
}

function finalize(reason?: string) {
	if (!currentStream || !out) return;
	if (!started) ensureStarted();
	currentStream.push({ type: "done", reason: reason === "length" ? "length" : "stop", message: out });
	currentStream.end();
	currentStream = null;
}

function endToolUse() {
	if (!currentStream || !out) return;
	out.stopReason = "toolUse";
	currentStream.push({ type: "done", reason: "toolUse", message: out });
	currentStream.end();
	currentStream = null;
}

function processEvent(msg: SDKMessage, toPiMap: Map<string, string>, model: Model<any>, rewriteSkills: boolean) {
	if (!currentStream || !out) return;
	sawStream = true;
	const ev = (msg as any).event;

	if (ev?.type === "message_start" && ev.message?.usage) { addUsage(out, ev.message.usage, model); return; }

	if (ev?.type === "content_block_start") {
		ensureStarted();
		const cb = ev.content_block;
		if (cb?.type === "text") {
			blocks.push({ type: "text", text: "", index: ev.index });
			currentStream.push({ type: "text_start", contentIndex: blocks.length - 1, partial: out });
		} else if (cb?.type === "thinking") {
			blocks.push({ type: "thinking", thinking: "", thinkingSignature: "", index: ev.index });
			currentStream.push({ type: "thinking_start", contentIndex: blocks.length - 1, partial: out });
		} else if (cb?.type === "tool_use") {
			sawTool = true;
			blocks.push({ type: "toolCall", id: cb.id, name: toPiName(cb.name, toPiMap), arguments: cb.input ?? {}, partialJson: "", index: ev.index });
			currentStream.push({ type: "toolcall_start", contentIndex: blocks.length - 1, partial: out });
		}
		return;
	}

	if (ev?.type === "content_block_delta") {
		const idx = blocks.findIndex((b: any) => b.index === ev.index);
		const b = blocks[idx];
		if (!b) return;
		const d = ev.delta;
		if (d?.type === "text_delta" && b.type === "text") {
			b.text += d.text;
			currentStream.push({ type: "text_delta", contentIndex: idx, delta: d.text, partial: out });
		} else if (d?.type === "thinking_delta" && b.type === "thinking") {
			b.thinking += d.thinking;
			currentStream.push({ type: "thinking_delta", contentIndex: idx, delta: d.thinking, partial: out });
		} else if (d?.type === "input_json_delta" && b.type === "toolCall") {
			b.partialJson += d.partial_json;
			b.arguments = tryJson(b.partialJson, b.arguments);
			currentStream.push({ type: "toolcall_delta", contentIndex: idx, delta: d.partial_json, partial: out });
		} else if (d?.type === "signature_delta" && b.type === "thinking") {
			b.thinkingSignature = (b.thinkingSignature ?? "") + d.signature;
		}
		return;
	}

	if (ev?.type === "content_block_stop") {
		const idx = blocks.findIndex((b: any) => b.index === ev.index);
		const b = blocks[idx];
		if (!b) return;
		delete b.index;
		if (b.type === "text") currentStream.push({ type: "text_end", contentIndex: idx, content: b.text, partial: out });
		else if (b.type === "thinking") currentStream.push({ type: "thinking_end", contentIndex: idx, content: b.thinking, partial: out });
		else if (b.type === "toolCall") {
			sawTool = true;
			b.arguments = mapArgs(b.name, tryJson(b.partialJson, b.arguments), rewriteSkills);
			delete b.partialJson;
			currentStream.push({ type: "toolcall_end", contentIndex: idx, toolCall: b, partial: out });
		}
		return;
	}

	if (ev?.type === "message_delta") {
		out.stopReason = ev.delta?.stop_reason === "tool_use" ? "toolUse" : ev.delta?.stop_reason === "max_tokens" ? "length" : "stop";
		if (ev.usage) addUsage(out, ev.usage, model);
		return;
	}

	if (ev?.type === "message_stop" && sawTool) { endToolUse(); return; }
}

function processAssistant(msg: SDKMessage, toPiMap: Map<string, string>, model: Model<any>, rewriteSkills: boolean) {
	if (sawStream) return;
	const content = (msg as any).message?.content;
	if (!content) return;
	for (const b of content) {
		if (b.type === "text" && b.text) {
			ensureStarted();
			blocks.push({ type: "text", text: b.text });
			const i = blocks.length - 1;
			currentStream?.push({ type: "text_start", contentIndex: i, partial: out });
			currentStream?.push({ type: "text_delta", contentIndex: i, delta: b.text, partial: out });
			currentStream?.push({ type: "text_end", contentIndex: i, content: b.text, partial: out });
		} else if (b.type === "thinking") {
			ensureStarted();
			blocks.push({ type: "thinking", thinking: b.thinking ?? "", thinkingSignature: b.signature ?? "" });
			const i = blocks.length - 1;
			currentStream?.push({ type: "thinking_start", contentIndex: i, partial: out });
			if (b.thinking) currentStream?.push({ type: "thinking_delta", contentIndex: i, delta: b.thinking, partial: out });
			currentStream?.push({ type: "thinking_end", contentIndex: i, content: b.thinking ?? "", partial: out });
		} else if (b.type === "tool_use") {
			ensureStarted();
			sawTool = true;
			const piName = toPiName(b.name, toPiMap);
			blocks.push({ type: "toolCall", id: b.id, name: piName, arguments: mapArgs(piName, b.input, rewriteSkills) });
			const i = blocks.length - 1;
			currentStream?.push({ type: "toolcall_start", contentIndex: i, partial: out });
			currentStream?.push({ type: "toolcall_end", contentIndex: i, toolCall: blocks[i], partial: out });
		}
	}
	if ((msg as any).message?.usage && out) addUsage(out, (msg as any).message.usage, model);
	if (sawTool) endToolUse();
}

async function consume(
	q: ReturnType<typeof query>, toPiMap: Map<string, string>, model: Model<any>,
	rewriteSkills: boolean, aborted: () => boolean,
): Promise<void> {
	for await (const msg of q) {
		if (aborted()) break;
		if (!currentStream || !out) continue;
		if (msg.type === "stream_event") processEvent(msg, toPiMap, model, rewriteSkills);
		else if (msg.type === "assistant") processAssistant(msg, toPiMap, model, rewriteSkills);
		else if (msg.type === "result" && !sawStream && msg.subtype === "success") {
			ensureStarted();
			const t = msg.result || "";
			blocks.push({ type: "text", text: t });
			const i = blocks.length - 1;
			currentStream?.push({ type: "text_start", contentIndex: i, partial: out });
			currentStream?.push({ type: "text_delta", contentIndex: i, delta: t, partial: out });
			currentStream?.push({ type: "text_end", contentIndex: i, content: t, partial: out });
		}
	}
}

// --- Provider entry point ---

function streamSimple(model: Model<any>, ctx: Context, options?: SimpleStreamOptions): AssistantMessageEventStream {
	const stream = createAssistantMessageEventStream();

	// Tool result delivery — resolve waiting MCP handlers
	if (activeQuery) {
		currentStream = stream;
		reset(model);
		for (const r of extractToolResults(ctx)) {
			if (pendingCalls.length) pendingCalls.shift()!.resolve(r);
			else pendingResults.push(r);
		}
		if (session) session.cursor = ctx.messages.length;
		return stream;
	}

	// Orphaned tool result (e.g. after abort)
	if (ctx.messages[ctx.messages.length - 1]?.role === "toolResult") {
		if (session) session.cursor = ctx.messages.length;
		queueMicrotask(() => { reset(model); stream.push({ type: "done", reason: "stop", message: out }); stream.end(); });
		return stream;
	}

	// --- Fresh query ---

	// Save parent state if reentrant (subagent starting a query while parent is suspended)
	const isReentrant = activeQuery !== null;
	if (isReentrant) {
		queryStateStack.push({ activeQuery, currentStream, pendingCalls: [...pendingCalls], pendingResults: [...pendingResults] });
	}

	currentStream = stream;
	pendingCalls.length = 0;
	pendingResults.length = 0;
	reset(model);

	const { tools, toSdk, toPi } = resolveMcpTools(ctx);
	const cwd = (options as any)?.cwd ?? process.cwd();
	const sdkModelId = toClaudeSdkModelId(model.id);
	const resumeId = syncSession(ctx.messages, cwd, toSdk, sdkModelId);
	const imgBlocks = userPromptBlocks(ctx.messages);
	const prompt: string | AsyncIterable<SDKUserMessage> = imgBlocks
		? wrapPromptStream(imgBlocks)
		: userPrompt(ctx.messages) || "[continue]";
	const effort = options?.reasoning ? EFFORT[options.reasoning] : undefined;

	// Build system prompt append from AGENTS.md + skills
	const agentsAppend = extractAgentsAppend();
	const skillsAppend = extractSkillsBlock(ctx.systemPrompt);
	const appendParts = [agentsAppend, skillsAppend].filter(Boolean);
	const systemPromptAppend = appendParts.length ? appendParts.join("\n\n") : undefined;
	const rewriteSkills = Boolean(skillsAppend);

	const q = query({
		prompt,
		options: {
			cwd,
			disallowedTools: DISALLOWED_BUILTIN_TOOLS,
			allowedTools: [`mcp__${MCP_SERVER_NAME}__*`],
			permissionMode: "bypassPermissions",
			includePartialMessages: true,
			systemPrompt: {
				type: "preset", preset: "claude_code",
				...(systemPromptAppend ? { append: systemPromptAppend } : {}),
			},
			extraArgs: { model: sdkModelId },
			...(effort ? { effort } : {}),
			...(tools.length ? { mcpServers: buildMcp(tools) } : {}),
			...(resumeId ? { resume: resumeId } : {}),
		},
	});
	activeQuery = q;

	let aborted = false;
	const onAbort = () => {
		aborted = true;
		for (const p of pendingCalls) p.resolve({ content: [{ type: "text", text: "Aborted" }] });
		pendingCalls.length = 0;
		pendingResults.length = 0;
		void q.interrupt().catch(() => {});
		try { q.close(); } catch {}
	};
	if (options?.signal?.aborted) onAbort();
	else options?.signal?.addEventListener("abort", onAbort, { once: true });

	consume(q, toPi, model, rewriteSkills, () => aborted)
		.then(() => {
			if (!aborted) {
				// Keep using locally rebuilt cc-session-io sessions as the source of truth.
				// Do NOT adopt the live SDK session_id here — those can become invalid when
				// switching providers/models in an existing pi session.
				if (session) session.cursor = Math.max(ctx.messages.length, session.cursor);
			}
			if (aborted || options?.signal?.aborted) {
				if (out) { out.stopReason = "aborted"; out.errorMessage = "Aborted"; }
				currentStream?.push({ type: "error", reason: "aborted", error: out! });
				currentStream?.end();
				currentStream = null;
			} else {
				finalize(out?.stopReason);
			}
		})
		.catch((err) => {
			if (out) { out.stopReason = "error"; out.errorMessage = err instanceof Error ? err.message : String(err); }
			currentStream?.push({ type: "error", reason: "error", error: out! });
			currentStream?.end();
			currentStream = null;
		})
		.finally(() => {
			options?.signal?.removeEventListener("abort", onAbort);
			if (activeQuery === q) {
				// Drain queues before restoring parent state
				for (const p of pendingCalls) p.resolve({ content: [{ type: "text", text: "Query ended" }] });
				pendingCalls.length = 0;
				pendingResults.length = 0;

				if (isReentrant && queryStateStack.length > 0) {
					const saved = queryStateStack.pop()!;
					activeQuery = saved.activeQuery;
					currentStream = saved.currentStream;
					pendingCalls.length = 0;
					pendingCalls.push(...saved.pendingCalls);
					pendingResults.length = 0;
					pendingResults.push(...saved.pendingResults);
				} else {
					activeQuery = null;
				}
			}
			q.close();
		});

	return stream;
}

// --- Registration ---

const REG_KEY = Symbol.for("pi-claude-sdk:registered");

export default function (pi: ExtensionAPI) {
	const g = globalThis as Record<symbol, any>;
	if (!g[REG_KEY]) {
		g[REG_KEY] = streamSimple;
		pi.registerProvider(PROVIDER_ID, {
			baseUrl: PROVIDER_ID, apiKey: "not-used", api: PROVIDER_ID,
			models: MODELS, streamSimple,
		});
	}

	const resetSessionState = () => {
		session = null;
		activeQuery = null;
	};
	const clear = () => {
		if (g[REG_KEY] === streamSimple) g[REG_KEY] = undefined;
		resetSessionState();
	};
	pi.on("session_start", resetSessionState);
	pi.on("session_shutdown", clear);
}
