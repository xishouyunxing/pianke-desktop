<script setup lang="ts">
import { computed, onMounted, onUnmounted, ref } from "vue";
import { invoke } from "@tauri-apps/api/core";

type BackendStatus = {
  running: boolean;
  healthy: boolean;
  url: string | null;
  token: string | null;
  port: number | null;
  python: string | null;
  backend_kind: string;
  backend_dir: string | null;
  message: string;
};

const status = ref<BackendStatus>({
  running: false,
  healthy: false,
  url: null,
  token: null,
  port: null,
  python: null,
  backend_kind: "rust-fast",
  backend_dir: null,
  message: "正在启动片刻引擎...",
});
const iframeEl = ref<HTMLIFrameElement | null>(null);
const selectedFolder = ref("");
const busy = ref(false);
let pollTimer: number | undefined;

const backendOrigin = computed(() => {
  if (!status.value.url) return "*";
  try {
    return new URL(status.value.url).origin;
  } catch {
    return "*";
  }
});

const frameUrl = computed(() => {
  if (!status.value.url) return "";
  const url = new URL(status.value.url);
  if (status.value.token) url.searchParams.set("token", status.value.token);
  url.searchParams.set("desktop", "1");
  return url.toString();
});

async function refreshStatus() {
  status.value = await invoke<BackendStatus>("backend_status");
}

async function restartBackend() {
  busy.value = true;
  try {
    status.value = await invoke<BackendStatus>("restart_backend");
  } finally {
    busy.value = false;
  }
}

async function chooseFolder() {
  const folder = await invoke<string | null>("pick_folder");
  if (!folder) return;
  selectedFolder.value = folder;
  postFolder(folder);
}

function postFolder(folder: string) {
  iframeEl.value?.contentWindow?.postMessage(
    { type: "pianke:set-folder", folder },
    backendOrigin.value,
  );
}

function postFolderPickResult(requestId: string, payload: { folder?: string | null; error?: string }) {
  iframeEl.value?.contentWindow?.postMessage(
    { type: "pianke:pick-folder-result", requestId, ...payload },
    "*",
  );
}

function postOpenUrlResult(requestId: string, payload: { ok?: boolean; error?: string }) {
  iframeEl.value?.contentWindow?.postMessage(
    { type: "pianke:open-url-result", requestId, ...payload },
    "*",
  );
}

function canTrustFrameMessage(event: MessageEvent) {
  const frame = iframeEl.value?.contentWindow;
  if (!frame) return false;
  if (event.source === frame) return true;
  const origin = backendOrigin.value;
  return origin !== "*" && event.origin === origin;
}

async function handleFrameMessage(event: MessageEvent) {
  const data = event.data || {};
  const allowed = data.type === "pianke:open-url" || data.type === "pianke:pick-folder" || data.type === "pianke:bridge-ready";
  if (!allowed) return;
  if (!canTrustFrameMessage(event)) return;

  if (data.type === "pianke:bridge-ready") {
    iframeEl.value?.contentWindow?.postMessage({ type: "pianke:bridge-ack" }, "*");
    return;
  }

  if (data.type === "pianke:open-url" && typeof data.requestId === "string" && typeof data.url === "string") {
    try {
      await invoke("open_external_url", { url: data.url });
      postOpenUrlResult(data.requestId, { ok: true });
    } catch (err) {
      postOpenUrlResult(data.requestId, {
        ok: false,
        error: err instanceof Error ? err.message : String(err),
      });
    }
    return;
  }

  if (data.type !== "pianke:pick-folder" || typeof data.requestId !== "string") return;

  try {
    const folder = await invoke<string | null>("pick_folder");
    if (folder) selectedFolder.value = folder;
    postFolderPickResult(data.requestId, { folder });
  } catch (err) {
    postFolderPickResult(data.requestId, {
      error: err instanceof Error ? err.message : String(err),
    });
  }
}

function handleFrameLoad() {
  if (selectedFolder.value) postFolder(selectedFolder.value);
}

onMounted(async () => {
  window.addEventListener("message", handleFrameMessage);
  await refreshStatus();
  pollTimer = window.setInterval(async () => {
    if (!status.value.healthy) await refreshStatus();
  }, 1200);
});

onUnmounted(() => {
  window.removeEventListener("message", handleFrameMessage);
  if (pollTimer) window.clearInterval(pollTimer);
});
</script>

<template>
  <main class="shell">
    <aside class="rail">
      <div class="brand">
        <span class="brand-mark"></span>
        <div>
          <strong>片刻</strong>
          <span>桌面版</span>
        </div>
      </div>

      <section class="panel">
        <p class="eyebrow">引擎状态</p>
        <h1>{{ status.healthy ? "引擎已就绪" : "正在唤醒引擎" }}</h1>
        <p class="copy">{{ status.message }}</p>
        <dl class="facts">
          <div>
            <dt>引擎</dt>
            <dd>{{ status.backend_kind === "rust-fast" ? "Rust Fast" : (status.python ?? "Python") }}</dd>
          </div>
        </dl>
      </section>

      <div class="actions">
        <button class="secondary" :disabled="busy" @click="restartBackend">
          {{ busy ? "重启中..." : "重启引擎" }}
        </button>
      </div>

      <p v-if="selectedFolder" class="folder">{{ selectedFolder }}</p>
    </aside>

    <section class="viewport">
      <iframe
        v-if="status.url"
        ref="iframeEl"
        :src="frameUrl"
        title="片刻 Web UI"
        @load="handleFrameLoad"
      />
      <div v-else class="empty">
        <h2>引擎未启动</h2>
        <p>请尝试重启引擎；正式安装包会启动内置 Rust 后端，不需要 Python。</p>
      </div>
    </section>
  </main>
</template>
