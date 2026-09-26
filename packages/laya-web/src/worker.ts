/**
 * Module Web Worker that hosts the ONNX session, so downloading ~850 MB of weights and running
 * a forward pass never blocks the page. Paired with `LayaWorkerClient` in client.ts.
 */
import { Laya, type LayaInfo, type LayaOptions } from "./laya.js";
import type { LoadProgress, Question, SystemOneResult } from "./types.js";

export type WorkerRequest =
  | {
      id: number;
      type: "load";
      opts: Omit<LayaOptions, "onProgress" | "sessionOptions">;
    }
  | {
      id: number;
      type: "predict";
      state: unknown;
      questions: Record<string, Question>;
    }
  | { id: number; type: "close" };

export type WorkerResponse =
  | { id: number; type: "progress"; progress: LoadProgress }
  | { id: number; type: "loaded"; info: LayaInfo }
  | { id: number; type: "result"; result: SystemOneResult }
  | { id: number; type: "closed" }
  | { id: number; type: "error"; message: string };

let laya: Laya | null = null;

const post = (msg: WorkerResponse) =>
  (self as unknown as Worker).postMessage(msg);

self.addEventListener("message", (event: MessageEvent<WorkerRequest>) => {
  void handle(event.data);
});

async function handle(req: WorkerRequest): Promise<void> {
  try {
    switch (req.type) {
      case "load": {
        laya = await Laya.load({
          ...req.opts,
          onProgress: (progress) =>
            post({ id: req.id, type: "progress", progress }),
        });
        post({ id: req.id, type: "loaded", info: laya.info });
        break;
      }
      case "predict": {
        if (!laya) throw new Error("model is not loaded");
        const result = await laya.predict(req.state, req.questions);
        post({ id: req.id, type: "result", result });
        break;
      }
      case "close": {
        await laya?.close();
        laya = null;
        post({ id: req.id, type: "closed" });
        break;
      }
    }
  } catch (err) {
    post({
      id: req.id,
      type: "error",
      message: err instanceof Error ? err.message : String(err),
    });
  }
}
