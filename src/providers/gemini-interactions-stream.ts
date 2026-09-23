import type { Response, ResponseOutputMessage, ResponseStreamEvent } from "../types.js";
import { ProviderError } from "../errors.js";
import { includeWhen, isNumber, isObject, isString, iterateClosing } from "../utils.js";
import {
  convertInteractionToResponse,
  isModelOutputStep,
  isTextContent,
  isThoughtStep,
  type GeminiInteraction,
  type GeminiInteractionStep,
} from "./gemini-interactions.js";

export interface GeminiInteractionStreamEvent {
  delta?: GeminiInteractionStreamDelta;
  error?: { code?: unknown; message?: unknown } | null;
  eventType?: unknown;
  event_type?: unknown;
  index?: unknown;
  interaction?: GeminiInteraction;
  step?: GeminiInteractionStep;
  type?: unknown;
}

export interface GeminiInteractionStreamDelta {
  raw?: unknown;
  text?: unknown;
  type?: unknown;
}

function eventType(event: GeminiInteractionStreamEvent): string {
  const value = event.event_type ?? event.eventType ?? event.type;
  return isString(value) ? value : "";
}

function deltaType(delta: GeminiInteractionStreamDelta): string {
  return isString(delta.type) ? delta.type : "";
}

function stepType(step: GeminiInteractionStep): string {
  return isString(step.type) ? step.type : "";
}

function raiseStreamError(message: string, code?: string): never {
  throw new ProviderError(message, {
    provider: "gemini",
    ...includeWhen(code !== undefined, { code }),
  });
}

function terminalEvent(response: Response, sequenceNumber: number): ResponseStreamEvent {
  if (response.status === "completed") {
    return { response, sequence_number: sequenceNumber, type: "response.completed" };
  }
  if (response.status === "failed") {
    return { response, sequence_number: sequenceNumber, type: "response.failed" };
  }
  return { response, sequence_number: sequenceNumber, type: "response.incomplete" };
}

function asInteraction(value: GeminiInteraction | undefined): GeminiInteraction {
  return value ?? {};
}

interface ConvertedStreamEvents {
  events: ResponseStreamEvent[];
  terminal: boolean;
}

function convertedEvents(events: ResponseStreamEvent[], terminal: boolean): ConvertedStreamEvents {
  return { events, terminal };
}

class TextStreamState {
  readonly model: string;
  interactionId = "";
  private readonly openSteps = new Set<number>();
  private sequence = 0;
  private started = false;
  private readonly textSteps = new Map<number, { outputIndex: number; text: string }>();

  constructor(model: string) {
    this.model = model;
  }

  convert(event: GeminiInteractionStreamEvent): ConvertedStreamEvents {
    const type = eventType(event);
    if (type === "interaction.created") {
      return convertedEvents(this.created(asInteraction(event.interaction)), false);
    }
    if (type === "step.start") return convertedEvents(this.stepStarted(event), false);
    if (type === "step.delta") return convertedEvents(this.stepDelta(event), false);
    if (type === "step.stop") return convertedEvents(this.stepStopped(event), false);
    if (type === "interaction.completed") {
      return convertedEvents([this.completed(asInteraction(event.interaction))], true);
    }
    if (type === "error") this.error(event);
    if (type === "UNKNOWN" || type === "unknown" || type.length === 0) {
      return convertedEvents([], false);
    }
    if (type === "interaction.status_update" || type === "status_update") {
      if (!this.started) {
        raiseStreamError(
          "Gemini interaction stream emitted a status update before interaction.created",
        );
      }
      return convertedEvents([], false);
    }
    return convertedEvents([], false);
  }

  incomplete(): never {
    raiseStreamError("Gemini interaction stream ended before interaction.completed");
  }

  private nextSequence(): number {
    const sequence = this.sequence;
    this.sequence += 1;
    return sequence;
  }

  private created(interaction: GeminiInteraction): ResponseStreamEvent[] {
    if (this.started) {
      raiseStreamError("Gemini interaction stream emitted interaction.created more than once");
    }
    this.started = true;
    const response = convertInteractionToResponse(interaction, this.model);
    this.interactionId = response.id;
    return [
      { response, sequence_number: this.nextSequence(), type: "response.created" },
      {
        response: { ...response, status: "in_progress" },
        sequence_number: this.nextSequence(),
        type: "response.in_progress",
      },
    ];
  }

  private stepIndex(event: GeminiInteractionStreamEvent): number {
    if (!isNumber(event.index) || !Number.isInteger(event.index)) {
      raiseStreamError("Gemini interaction stream emitted a step without an integer index");
    }
    return event.index;
  }

  private stepStarted(event: GeminiInteractionStreamEvent): ResponseStreamEvent[] {
    if (!this.started) {
      raiseStreamError("Gemini interaction stream emitted step.start before interaction.created");
    }
    const index = this.stepIndex(event);
    if (this.openSteps.has(index) || this.textSteps.has(index)) {
      raiseStreamError(`Gemini interaction stream started step ${index} more than once`);
    }
    this.openSteps.add(index);
    const step = event.step ?? {};
    if (isThoughtStep(step)) return [];
    if (stepType(step) === "UNKNOWN" || stepType(step) === "unknown") return [];
    if (!isModelOutputStep(step)) return [];

    const content = step.content ?? [];
    if (content.some((part) => !isTextContent(part))) {
      raiseStreamError("Gemini interaction stream returned non-text model output");
    }
    const prefix = content
      .flatMap((part) => (isTextContent(part) && isString(part.text) ? [part.text] : []))
      .join("");
    const outputIndex = this.textSteps.size;
    this.textSteps.set(index, { outputIndex, text: prefix });
    const itemId = `msg-${this.interactionId}-${outputIndex}`;
    const events: ResponseStreamEvent[] = [
      {
        item: {
          content: [],
          id: itemId,
          role: "assistant",
          status: "in_progress",
          type: "message",
        },
        output_index: outputIndex,
        sequence_number: this.nextSequence(),
        type: "response.output_item.added",
      },
      {
        content_index: 0,
        item_id: itemId,
        output_index: outputIndex,
        part: { annotations: [], text: "", type: "output_text" },
        sequence_number: this.nextSequence(),
        type: "response.content_part.added",
      },
    ];
    if (prefix.length > 0) events.push(this.textDelta(index, prefix));
    return events;
  }

  private stepDelta(event: GeminiInteractionStreamEvent): ResponseStreamEvent[] {
    if (!this.started) {
      raiseStreamError("Gemini interaction stream emitted step.delta before interaction.created");
    }
    const index = this.stepIndex(event);
    if (!this.openSteps.has(index)) {
      raiseStreamError(
        `Gemini interaction stream emitted a delta before step.start for step ${index}`,
      );
    }
    if (!this.textSteps.has(index)) return [];
    const delta = event.delta ?? {};
    const type = deltaType(delta);
    if (type === "UNKNOWN" || type === "unknown") return [];
    if (type === "text_annotation" || type === "thought_signature") return [];
    if (type !== "text" && type.length > 0) {
      raiseStreamError("Gemini interaction stream returned non-text model output delta");
    }
    if (!isString(delta.text)) {
      raiseStreamError("Gemini interaction stream returned non-text model output delta");
    }
    const current = this.textSteps.get(index);
    if (current === undefined) return [];
    this.textSteps.set(index, {
      outputIndex: current.outputIndex,
      text: current.text + delta.text,
    });
    return [this.textDelta(index, delta.text)];
  }

  private textDelta(stepIndex: number, text: string): ResponseStreamEvent {
    const current = this.textSteps.get(stepIndex);
    const outputIndex = current?.outputIndex ?? 0;
    return {
      content_index: 0,
      delta: text,
      item_id: `msg-${this.interactionId}-${outputIndex}`,
      logprobs: [],
      output_index: outputIndex,
      sequence_number: this.nextSequence(),
      type: "response.output_text.delta",
    };
  }

  private stepStopped(event: GeminiInteractionStreamEvent): ResponseStreamEvent[] {
    if (!this.started) {
      raiseStreamError("Gemini interaction stream emitted step.stop before interaction.created");
    }
    const index = this.stepIndex(event);
    if (!this.openSteps.has(index)) {
      raiseStreamError(`Gemini interaction stream stopped unknown step ${index}`);
    }
    this.openSteps.delete(index);
    const current = this.textSteps.get(index);
    if (current === undefined) return [];
    const itemId = `msg-${this.interactionId}-${current.outputIndex}`;
    const completedPart = {
      annotations: [],
      text: current.text,
      type: "output_text" as const,
    };
    const completedItem: ResponseOutputMessage = {
      content: [completedPart],
      id: itemId,
      role: "assistant",
      status: "completed",
      type: "message",
    };
    return [
      {
        content_index: 0,
        item_id: itemId,
        logprobs: [],
        output_index: current.outputIndex,
        sequence_number: this.nextSequence(),
        text: current.text,
        type: "response.output_text.done",
      },
      {
        content_index: 0,
        item_id: itemId,
        output_index: current.outputIndex,
        part: completedPart,
        sequence_number: this.nextSequence(),
        type: "response.content_part.done",
      },
      {
        item: completedItem,
        output_index: current.outputIndex,
        sequence_number: this.nextSequence(),
        type: "response.output_item.done",
      },
    ];
  }

  private completed(interaction: GeminiInteraction): ResponseStreamEvent {
    if (!this.started) {
      raiseStreamError("Gemini interaction stream completed before interaction.created");
    }
    if (this.openSteps.size > 0) {
      raiseStreamError(
        `Gemini interaction stream completed before step.stop for step ${Math.min(...this.openSteps)}`,
      );
    }
    const rebuiltSteps: GeminiInteractionStep[] = [...this.textSteps.values()]
      .toSorted((left, right) => left.outputIndex - right.outputIndex)
      .map((step) => ({
        content: [{ text: step.text, type: "text" }],
        type: "model_output",
      }));
    const response = convertInteractionToResponse(
      {
        ...interaction,
        id: this.interactionId,
        ...includeSteps(rebuiltSteps),
      },
      this.model,
    );
    for (const [index, item] of response.output.entries()) {
      if (index < this.textSteps.size && item.type === "message") item.status = "completed";
    }
    return terminalEvent(response, this.nextSequence());
  }

  private error(event: GeminiInteractionStreamEvent): never {
    const message =
      event.error !== undefined &&
      event.error !== null &&
      isString(event.error.message) &&
      event.error.message.length > 0
        ? event.error.message
        : "Gemini interaction failed";
    const code =
      event.error !== undefined && event.error !== null && isString(event.error.code)
        ? event.error.code
        : undefined;
    raiseStreamError(message, code);
  }
}

function includeSteps(steps: GeminiInteractionStep[]): Pick<GeminiInteraction, "steps"> | object {
  return steps.length === 0 ? {} : { steps };
}

export async function* convertInteractionStream(
  stream: AsyncIterable<unknown>,
  model: string,
): AsyncIterable<ResponseStreamEvent> {
  const state = new TextStreamState(model);
  for await (const raw of iterateClosing(stream)) {
    const event = isObject(raw) ? raw : {};
    const { events, terminal } = state.convert(event);
    for (const converted of events) yield converted;
    if (terminal) return;
  }
  state.incomplete();
}
