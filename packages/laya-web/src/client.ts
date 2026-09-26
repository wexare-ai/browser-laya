/**
 * Main-thread handle to a Laya instance running inside a Web Worker. The surface matches the
 * in-thread `Laya` class, so swapping between them is a one-line change.
 */
import type { LayaInfo, LayaOptions } from "./laya.js";
import type { LoadProgress, Question, SystemOneResult } from "./types.js";
import type { WorkerRequest, WorkerResponse } from "./worker.js";

export interface WorkerClientOptions extends Omit<
  LayaOptions,
  "sessionOptions"
> {
  /**
   * The worker to drive. Bundlers need the `new Worker(new URL(...), {type:"module"})` form
   * written literally at the call site, so the caller supplies the worker.
   */
  worker: Worker;
}

/** Omit that distributes over a discriminated union instead of collapsing it. */
type DistributiveOmit<T, K extends keyof never> = T extends unknown
  ? Omit<T, K>
  : never;

type OutgoingRequest = DistributiveOmit<WorkerRequest, "id">;

interface Pending {
  resolve: (value: unknown) => void;
  reject: (reason: Error) => void;
  onProgress?: (p: LoadProgress) => void;
}

export class LayaWorkerClient {
  private nextId = 1;
  private readonly pending = new Map<number, Pending>();
  /** populated by `load`; describes the bundle and device actually in use */
  info!: LayaInfo;

  private constructor(private readonly worker: Worker) {
    worker.addEventListener("message", (event: MessageEvent<WorkerResponse>) =>
      this.onMessage(event.data),
    );
    worker.addEventListener("error", (event) =>
      this.failAll(new Error(`worker failed: ${event.message}`)),
    );
  }

  static async load(opts: WorkerClientOptions): Promise<LayaWorkerClient> {
    const { worker, onProgress, ...rest } = opts;
    const client = new LayaWorkerClient(worker);
    client.info = await client.send<LayaInfo>(
      { type: "load", opts: rest },
      onProgress,
    );
    return client;
  }

  private onMessage(msg: WorkerResponse): void {
    const entry = this.pending.get(msg.id);
    if (!entry) return;
    if (msg.type === "progress") {
      entry.onProgress?.(msg.progress);
      return;
    }
    this.pending.delete(msg.id);
    if (msg.type === "error") entry.reject(new Error(msg.message));
    else if (msg.type === "loaded") entry.resolve(msg.info);
    else if (msg.type === "result") entry.resolve(msg.result);
    else entry.resolve(undefined);
  }

  private failAll(err: Error): void {
    for (const [, entry] of this.pending) entry.reject(err);
    this.pending.clear();
  }

  private send<T>(
    req: OutgoingRequest,
    onProgress?: (p: LoadProgress) => void,
  ): Promise<T> {
    const id = this.nextId++;
    return new Promise<T>((resolve, reject) => {
      this.pending.set(id, {
        resolve: resolve as Pending["resolve"],
        reject,
        onProgress,
      });
      this.worker.postMessage({ ...req, id } as WorkerRequest);
    });
  }

  predict<Q extends Record<string, Question>>(
    state: unknown,
    questions: Q,
  ): Promise<SystemOneResult<Q>> {
    return this.send<SystemOneResult<Q>>({ type: "predict", state, questions });
  }

  async close(): Promise<void> {
    await this.send<void>({ type: "close" });
    this.worker.terminate();
  }
}
