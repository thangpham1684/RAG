/* ============================================================
   RAG Enterprise Pro — Application Logic
   ============================================================ */

const API_BASE_URL = "http://localhost:8000";

// ----- DOM Refs -----
const DOM = {
    sidebar: document.getElementById("sidebar"),
    sidebarOverlay: document.getElementById("sidebar-overlay"),
    sidebarToggle: document.getElementById("sidebar-toggle"),
    sidebarClose: document.getElementById("sidebar-close"),

    apiStatus: document.getElementById("api-status"),
    dbStatus: document.getElementById("db-status"),
    filesCountValue: document.getElementById("files-count-value"),

    fileList: document.getElementById("file-list"),
    fileSection: document.getElementById("file-section"),
    fileFilterInfo: document.getElementById("file-filter-info"),
    fileFilterCount: document.getElementById("file-filter-count"),
    btnRefreshFiles: document.getElementById("btn-refresh-files"),
    btnClearFiles: document.getElementById("btn-clear-files"),

    btnReindex: document.getElementById("btn-reindex"),
    ingestionProgress: document.getElementById("ingestion-progress"),
    progressFill: document.getElementById("progress-fill"),
    progressText: document.getElementById("progress-text"),
    indexMessage: document.getElementById("index-message"),

    evidenceToggle: document.getElementById("evidence-toggle"),

    // API Key management
    apiKeysPanel: document.getElementById("api-keys-panel"),
    apiKeysList: document.getElementById("api-keys-list"),
    apiKeysCount: document.getElementById("api-keys-count"),
    apiKeysError: document.getElementById("api-keys-error"),
    btnCreateKey: document.getElementById("btn-create-key"),
    createKeyForm: document.getElementById("create-key-form"),
    newKeyName: document.getElementById("new-key-name"),
    btnConfirmCreateKey: document.getElementById("btn-confirm-create-key"),
    btnCancelCreateKey: document.getElementById("btn-cancel-create-key"),
    newKeyDisplay: document.getElementById("new-key-display"),
    newKeyValue: document.getElementById("new-key-value"),
    btnCopyKey: document.getElementById("btn-copy-key"),
    btnDismissNewKey: document.getElementById("btn-dismiss-new-key"),

    chatForm: document.getElementById("chat-form"),
    chatInput: document.getElementById("chat-input"),
    btnSend: document.getElementById("btn-send"),
    chatHistory: document.getElementById("chat-history"),
    emptyState: document.getElementById("empty-state"),
    btnClear: document.getElementById("btn-clear-chat"),
};

// ----- State -----
let isApiOnline = false;
let selectedFiles = new Set();
let allFiles = [];
let conversationHistory = [];
const MAX_HISTORY_TURNS = 20; // Max (user+assistant) message pairs to keep
let _lastStatus = { api: null, db: null }; // Cache for health check to avoid unnecessary DOM updates

// ----- File Upload Helpers -----
const uploadArea = document.getElementById("upload-area");
const fileInput = document.getElementById("file-input");

function setupUpload() {
    // Click to open file picker
    uploadArea.addEventListener("click", (e) => {
        // Don't trigger if clicking message (to allow copy)
        if (e.target.closest(".upload-message") || e.target.closest(".upload-progress")) return;
        fileInput.click();
    });

    // Drag & drop
    uploadArea.addEventListener("dragover", (e) => {
        e.preventDefault();
        uploadArea.classList.add("drag-over");
    });
    uploadArea.addEventListener("dragleave", () => {
        uploadArea.classList.remove("drag-over");
    });
    uploadArea.addEventListener("drop", (e) => {
        e.preventDefault();
        uploadArea.classList.remove("drag-over");
        if (e.dataTransfer.files.length > 0) {
            handleFiles(e.dataTransfer.files);
        }
    });

    // File input change
    fileInput.addEventListener("change", () => {
        if (fileInput.files.length > 0) {
            handleFiles(fileInput.files);
            fileInput.value = ""; // Reset for re-upload
        }
    });
}

async function handleFiles(files) {
    const progressFill = document.getElementById("upload-progress-fill");
    const progressText = document.getElementById("upload-progress-text");
    const progressDiv = document.getElementById("upload-progress");
    const msgDiv = document.getElementById("upload-message");

    let successCount = 0;
    let failCount = 0;
    const total = files.length;

    progressDiv.style.display = "block";
    msgDiv.innerHTML = "";

    // Phase 1: Upload all files sequentially
    for (let i = 0; i < total; i++) {
        const file = files[i];
        const pct = Math.round(((i) / total) * 100);
        progressFill.style.width = pct + "%";
        progressText.textContent = `Đang tải ${i + 1}/${total}: ${escapeHtml(file.name)}`;

        try {
            const formData = new FormData();
            formData.append("file", file);

            const res = await fetch(`${API_BASE_URL}/api/v1/upload`, {
                method: "POST",
                body: formData,
            });

            const data = await res.json();
            if (res.ok) {
                successCount++;
            } else {
                msgDiv.innerHTML += `<div class="error"><i class="fas fa-exclamation-circle"></i> ${escapeHtml(file.name)}: ${escapeHtml(data.detail || res.statusText)}</div>`;
                failCount++;
            }
        } catch (err) {
            msgDiv.innerHTML += `<div class="error"><i class="fas fa-exclamation-circle"></i> ${escapeHtml(file.name)}: ${escapeHtml(err.message)}</div>`;
            failCount++;
        }
    }

    // Refresh file list right away (files are saved even before ingestion finishes)
    await fetchFiles();
    await checkHealth();

    // Phase 2: If any files uploaded, start a SINGLE ingestion job
    if (successCount > 0) {
        msgDiv.innerHTML = `<div class="success"><i class="fas fa-check-circle"></i> Đã tải ${successCount}/${total} file thành công</div>` + msgDiv.innerHTML;
        progressFill.style.width = "50%";
        progressText.textContent = "Ingestion đang chạy nền...";

        try {
            // Start a single ingestion job for all uploaded files
            const indexRes = await fetch(`${API_BASE_URL}/api/v1/index`, { method: "POST" });
            if (indexRes.ok) {
                const indexData = await indexRes.json();
                const jobId = indexData.job_id;

                if (jobId) {
                    const ingestResult = await pollIngestionStatus(jobId, 1500, 300000);

                    if (ingestResult.status === "success") {
                        progressFill.style.width = "100%";
                        progressText.textContent = "Ingestion hoàn tất!";
                        msgDiv.innerHTML = `<div class="success"><i class="fas fa-check-circle"></i> Đã index: ${ingestResult.total_nodes || 0} nodes</div>` + msgDiv.innerHTML;
                        // Refresh again to get updated counts
                        await fetchFiles();
                        await checkHealth();
                    } else if (ingestResult.status === "error") {
                        msgDiv.innerHTML += `<div class="error"><i class="fas fa-exclamation-circle"></i> Ingestion lỗi: ${escapeHtml(ingestResult.error || "Không xác định")}</div>`;
                    }
                }
            } else {
                const errData = await indexRes.json().catch(() => ({ detail: indexRes.statusText }));
                msgDiv.innerHTML += `<div class="warning"><i class="fas fa-info-circle"></i> ${escapeHtml(errData.detail || "Không thể khởi động ingestion")}</div>`;
            }
        } catch (err) {
            msgDiv.innerHTML += `<div class="warning"><i class="fas fa-info-circle"></i> Lỗi khi khởi động ingestion: ${escapeHtml(err.message)}</div>`;
        }
    }

    setTimeout(() => {
        progressDiv.style.display = "none";
        progressFill.style.width = "0%";
        if (successCount === total) {
            msgDiv.innerHTML = "";
        }
    }, 6000);
}

async function deleteFile(fileName) {
    if (!confirm(`Xoá tệp "${fileName}" và tất cả dữ liệu index liên quan?`)) {
        return;
    }

    try {
        const res = await fetch(`${API_BASE_URL}/api/v1/files/${encodeURIComponent(fileName)}`, {
            method: "DELETE",
        });

        if (!res.ok) {
            const err = await res.json().catch(() => ({ detail: res.statusText }));
            alert(`Lỗi khi xoá: ${err.detail || "Không xác định"}`);
            return;
        }

        const data = await res.json();
        // Refresh after deletion
        await fetchFiles();
        await checkHealth();

        // Show brief success in filter info
        DOM.fileFilterInfo.textContent = `Đã xoá "${fileName}" (${data.nodes_deleted || 0} nodes)`;
        setTimeout(() => updateFileFilterInfo(), 3000);
    } catch (err) {
        alert(`Lỗi kết nối khi xoá file: ${err.message}`);
    }
}

// ----- Init -----
document.addEventListener("DOMContentLoaded", () => {
    setupUpload();

    // Sidebar toggle (mobile)
    DOM.sidebarToggle.addEventListener("click", () => {
        DOM.sidebar.classList.add("open");
        DOM.sidebarOverlay.classList.add("active");
    });
    DOM.sidebarClose.addEventListener("click", closeSidebar);
    DOM.sidebarOverlay.addEventListener("click", closeSidebar);

    // Textarea auto-resize
    DOM.chatInput.addEventListener("input", onInputChange);

    // Shift+Enter = newline, Enter = send
    DOM.chatInput.addEventListener("keydown", (e) => {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            if (!DOM.btnSend.disabled) {
                DOM.chatForm.dispatchEvent(new Event("submit", { cancelable: true }));
            }
        }
    });

    // Chat form submit
    DOM.chatForm.addEventListener("submit", onChatSubmit);

    // Clear chat
    DOM.btnClear.addEventListener("click", clearChat);

    // Reindex
    DOM.btnReindex.addEventListener("click", onReindex);

    // File actions
    DOM.btnRefreshFiles.addEventListener("click", fetchFiles);
    DOM.btnClearFiles.addEventListener("click", clearFileFilter);

    // API Key management
    DOM.btnCreateKey.addEventListener("click", showCreateKeyForm);
    DOM.btnConfirmCreateKey.addEventListener("click", onCreateKey);
    DOM.btnCancelCreateKey.addEventListener("click", cancelCreateKey);
    DOM.btnDismissNewKey.addEventListener("click", dismissNewKey);
    DOM.btnCopyKey.addEventListener("click", copyNewKey);

    // Fetch keys on load
    fetchApiKeys();

    // Evidence toggle — value read at render time in onChatSubmit

    // Enter key in create form
    DOM.newKeyName.addEventListener("keydown", (e) => {
        if (e.key === "Enter") {
            e.preventDefault();
            if (!DOM.btnConfirmCreateKey.disabled) {
                DOM.btnConfirmCreateKey.click();
            }
        }
    });

    // Chip suggestions
    document.querySelectorAll(".chip[data-prompt]").forEach((chip) => {
        chip.addEventListener("click", () => {
            DOM.chatInput.value = chip.dataset.prompt;
            DOM.chatInput.dispatchEvent(new Event("input"));
            DOM.chatInput.focus();
        });
    });

    // Start polling
    checkHealth();
    setInterval(checkHealth, 10000);
    fetchFiles();
});

function closeSidebar() {
    DOM.sidebar.classList.remove("open");
    DOM.sidebarOverlay.classList.remove("active");
}

// ----- Input -----
function onInputChange() {
    this.style.height = "auto";
    this.style.height = this.scrollHeight + "px";
    updateSendButton();
}

function updateSendButton() {
    const hasText = DOM.chatInput.value.trim().length > 0;
    DOM.btnSend.disabled = !(hasText && isApiOnline);
}

// ----- Health Check -----
async function checkHealth() {
    try {
        const res = await fetch(`${API_BASE_URL}/health`);
        const data = await res.json();
        isApiOnline = true;

        const apiState = "online";
        const dbState = data.db_loaded ? "online" : "warning";

        // Only update DOM if status actually changed
        if (apiState !== _lastStatus.api) {
            setStatus(DOM.apiStatus, apiState, "API: Sẵn sàng");
            _lastStatus.api = apiState;
        }
        if (dbState !== _lastStatus.db) {
            const dbLabel = data.db_loaded ? "Dữ liệu: Đã nạp" : "Dữ liệu: Chưa nạp";
            setStatus(DOM.dbStatus, dbState, dbLabel);
            _lastStatus.db = dbState;
        }

        DOM.chatInput.disabled = false;
        updateSendButton();
    } catch {
        isApiOnline = false;
        const offState = "offline";
        if (offState !== _lastStatus.api) {
            setStatus(DOM.apiStatus, offState, "API: Mất kết nối");
            _lastStatus.api = offState;
        }
        if (offState !== _lastStatus.db) {
            setStatus(DOM.dbStatus, offState, "Dữ liệu: Không xác định");
            _lastStatus.db = offState;
        }
        DOM.chatInput.disabled = true;
        DOM.btnSend.disabled = true;
    }
}

function setStatus(el, state, text) {
    const dot = el.querySelector(".status-dot");
    if (dot) {
        dot.className = "status-dot " + state;
    }
    const span = el.querySelector("span");
    if (span) span.textContent = text;
}

// ----- Files -----
async function fetchFiles() {
    try {
        const res = await fetch(`${API_BASE_URL}/api/v1/files`);
        const data = await res.json();
        allFiles = data.files || [];
    } catch {
        allFiles = [];
    }

    // Update count badge in status
    DOM.filesCountValue.textContent = allFiles.length;

    // Render file list
    renderFileList();
}

function renderFileList() {
    DOM.fileList.innerHTML = "";
    if (allFiles.length === 0) {
        DOM.fileList.innerHTML = '<div class="file-item" style="cursor:default;color:var(--text-muted);">Không có tài liệu</div>';
        return;
    }

    allFiles.forEach((f) => {
        const ext = f.name.split(".").pop().toLowerCase();
        const iconMap = { pdf: "fa-file-pdf", docx: "fa-file-word", xlsx: "fa-file-excel", txt: "fa-file-lines", md: "fa-file-lines" };
        const icon = iconMap[ext] || "fa-file";

        const item = document.createElement("div");
        item.className = "file-item" + (selectedFiles.has(f.name) ? " selected" : "");
        item.innerHTML = `
            <div class="file-checkbox"></div>
            <i class="fas ${icon} file-icon"></i>
            <span class="file-name" title="${escapeHtml(f.name)}">${escapeHtml(f.name)}</span>
            <button class="file-delete" title="Xoá tệp này" data-file="${escapeHtml(f.name)}">
                <i class="fas fa-trash-can"></i>
            </button>
        `;
        item.addEventListener("click", (e) => {
            // Don't toggle when clicking delete button
            if (e.target.closest(".file-delete")) return;
            toggleFile(f.name);
        });
        // Delete handler
        item.querySelector(".file-delete").addEventListener("click", (e) => {
            e.stopPropagation();
            deleteFile(f.name);
        });
        DOM.fileList.appendChild(item);
    });

    updateFileFilterInfo();
}

function toggleFile(name) {
    if (selectedFiles.has(name)) {
        selectedFiles.delete(name);
    } else {
        selectedFiles.add(name);
    }
    renderFileList();
}

function clearFileFilter() {
    selectedFiles.clear();
    renderFileList();
    DOM.fileFilterInfo.textContent = "";
}

function updateFileFilterInfo() {
    DOM.fileFilterCount.textContent = selectedFiles.size;
    if (selectedFiles.size > 0) {
        DOM.fileFilterInfo.textContent = `Đã chọn ${selectedFiles.size} file — chỉ tìm trong các file này`;
    } else {
        DOM.fileFilterInfo.textContent = "";
    }
}

// ----- Ingestion (Background) -----
async function pollIngestionStatus(jobId, intervalMs = 1000, maxDurationMs = 300000) {
    const startTime = Date.now();
    while (Date.now() - startTime < maxDurationMs) {
        try {
            const res = await fetch(`${API_BASE_URL}/api/v1/ingestion/status/${jobId}`);
            if (!res.ok) {
                if (res.status === 404) {
                    // Job might have been cleaned up — assume success if we see files
                    return { status: "success", message: "Job completed (cleaned up)" };
                }
                throw new Error(`Status check failed: ${res.statusText}`);
            }
            const data = await res.json();
            if (data.status === "success") {
                return data;
            }
            if (data.status === "error") {
                return data;
            }
        } catch (err) {
            if (Date.now() - startTime > 5000) {
                // After 5s, network errors are likely real
                throw err;
            }
            // Early network glitch — retry
        }
        await new Promise(r => setTimeout(r, intervalMs));
    }
    return { status: "timeout", message: "Quá thời gian chờ. Kiểm tra lại sau." };
}

async function onReindex() {
    DOM.btnReindex.disabled = true;
    DOM.btnReindex.innerHTML = '<i class="fas fa-circle-notch fa-spin"></i> Đang xây dựng...';
    DOM.ingestionProgress.style.display = "block";
    DOM.progressFill.style.width = "30%";
    DOM.progressText.textContent = "Đang xử lý tài liệu...";
    DOM.indexMessage.innerHTML = "";

    try {
        const res = await fetch(`${API_BASE_URL}/api/v1/index`, { method: "POST" });

        if (!res.ok) {
            const errBody = await res.json().catch(() => ({ detail: res.statusText }));
            throw new Error(errBody.detail || "Unknown error");
        }

        const startData = await res.json();
        const jobId = startData.job_id;

        DOM.progressFill.style.width = "50%";
        DOM.progressText.textContent = "Ingestion đang chạy nền...";
        DOM.indexMessage.innerHTML = `<span style="color:var(--text-muted)"><i class="fas fa-spinner fa-pulse"></i> Job: ${escapeHtml(jobId.slice(0, 8))}...</span>`;

        // Poll for completion
        const result = await pollIngestionStatus(jobId);

        if (result.status === "success") {
            DOM.progressFill.style.width = "100%";
            DOM.progressText.textContent = "Hoàn tất!";
            const total = result.total_nodes || 0;
            DOM.indexMessage.innerHTML = `<span style="color:var(--green)"><i class="fas fa-check-circle"></i> Index xong: ${total} nodes</span>`;
        } else if (result.status === "error") {
            throw new Error(result.error || "Ingestion thất bại");
        } else {
            DOM.indexMessage.innerHTML = `<span style="color:var(--yellow)"><i class="fas fa-clock"></i> ${escapeHtml(result.message || "Đang xử lý...")}</span>`;
            DOM.progressFill.style.width = "50%";
            DOM.progressText.textContent = "Đang chờ...";
        }

        // Refresh files & health
        await fetchFiles();
        await checkHealth();
    } catch (err) {
        DOM.indexMessage.innerHTML = `<span style="color:var(--red)"><i class="fas fa-exclamation-circle"></i> ${escapeHtml(err.message)}</span>`;
        DOM.progressFill.style.width = "0%";
        DOM.progressText.textContent = "Thất bại";
    }

    setTimeout(() => {
        DOM.ingestionProgress.style.display = "none";
        DOM.progressFill.style.width = "0%";
    }, 4000);

    DOM.btnReindex.innerHTML = '<i class="fas fa-sync-alt"></i> Xây dựng index';
    DOM.btnReindex.disabled = false;
}

// ----- Chat -----
async function onChatSubmit(e) {
    e.preventDefault();
    const query = DOM.chatInput.value.trim();
    if (!query) return;

    // Hide empty state
    DOM.emptyState.style.display = "none";

    // 1. Render user message
    const userContent = createMessage("user");
    userContent.textContent = query;

    // 2. Add user query to conversation history
    conversationHistory.push({ role: "user", content: query });

    // 3. Reset input
    DOM.chatInput.value = "";
    DOM.chatInput.style.height = "auto";
    DOM.chatInput.disabled = true;
    DOM.btnSend.disabled = true;

    // 4. Render bot placeholder
    const botContent = createMessage("bot");
    botContent.innerHTML = '<span class="cursor-blink"></span>';
    scrollToBottom();

    // 5. Build request with conversation history
    const body = { query };
    if (selectedFiles.size > 0) {
        body.selected_files = Array.from(selectedFiles);
    }
    if (conversationHistory.length > 0) {
        body.conversation_history = conversationHistory;
    }

    try {
        const res = await fetch(`${API_BASE_URL}/api/v1/chat`, {
            method: "POST",
            headers: { "Content-Type": "application/json", Accept: "text/plain" },
            body: JSON.stringify(body),
        });

        if (!res.ok) {
            const errData = await res.json().catch(() => ({}));
            botContent.innerHTML = `<i class="fas fa-exclamation-triangle" style="color:var(--red)"></i> Lỗi: ${escapeHtml(errData.detail || res.statusText)}`;
            return;
        }

        // 6. Stream response
        const reader = res.body.getReader();
        const decoder = new TextDecoder("utf-8");
        let accumulated = "";

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            const chunk = decoder.decode(value, { stream: true });
            accumulated += chunk;

            // Check for evidence header
            let displayText = accumulated;
            let evidenceHtml = "";
            if (accumulated.startsWith("[EVIDENCE:")) {
                const endIdx = accumulated.indexOf("]\n");
                if (endIdx !== -1) {
                    const evTag = accumulated.substring(0, endIdx + 1);
                    displayText = accumulated.substring(endIdx + 2);
                    evidenceHtml = buildEvidenceHtml(evTag);
                }
            }

            // Render markdown + evidence (batched via requestAnimationFrame)
            if (displayText.trim()) {
                if (!botContent._pendingRender) {
                    // First chunk in this frame — schedule render
                    botContent._pendingData = displayText;
                    botContent._pendingEvidence = evidenceHtml;
                    botContent._pendingRender = requestAnimationFrame(() => {
                        botContent._pendingRender = null;
                        const data = botContent._pendingData;
                        const evHtml = botContent._pendingEvidence || "";
                        botContent._pendingData = null;
                        botContent._pendingEvidence = null;
                        const rawHtml = marked.parse(data);
                        const safeHtml = DOMPurify.sanitize(rawHtml);
                        botContent.innerHTML = safeHtml + evHtml + '<span class="cursor-blink"></span>';

                        // Highlight code blocks
                        botContent.querySelectorAll("pre code").forEach((block) => {
                            try { hljs.highlightElement(block); } catch {}
                        });

                        // Scroll instantly after DOM update
                        if (DOM.chatHistory.scrollHeight - DOM.chatHistory.scrollTop - DOM.chatHistory.clientHeight < 150) {
                            DOM.chatHistory.scrollTop = DOM.chatHistory.scrollHeight;
                        }
                    });
                } else {
                    // Subsequent chunk in same frame — just update pending data
                    botContent._pendingData = displayText;
                    botContent._pendingEvidence = evidenceHtml;
                }
            }
        }

        // Done: remove cursor blink
        const finalDisplay = accumulated.startsWith("[EVIDENCE:") ? accumulated.substring(accumulated.indexOf("]\n") + 2) : accumulated;
        let finalEvidence = "";
        if (accumulated.startsWith("[EVIDENCE:")) {
            const endIdx = accumulated.indexOf("]\n");
            const evTag = accumulated.substring(0, endIdx + 1);
            finalEvidence = buildEvidenceHtml(evTag);
        }
        const finalHtml = DOMPurify.sanitize(marked.parse(finalDisplay));
        botContent.innerHTML = finalHtml + finalEvidence;
        botContent.querySelectorAll("pre code").forEach((block) => {
            try { hljs.highlightElement(block); } catch {}
        });

        // Add assistant response to conversation history
        conversationHistory.push({ role: "assistant", content: finalDisplay });
        // Trim history to prevent context overflow
        if (conversationHistory.length > MAX_HISTORY_TURNS) {
            conversationHistory = conversationHistory.slice(-MAX_HISTORY_TURNS);
        }
    } catch (err) {
        botContent.innerHTML = `<i class="fas fa-wifi" style="color:var(--red)"></i> Mất kết nối: ${escapeHtml(err.message)}`;
    }

    DOM.chatInput.disabled = false;
    DOM.chatInput.focus();
    scrollToBottom();
}

// ----- Helpers -----
function createMessage(role) {
    const wrapper = document.createElement("div");
    wrapper.className = "message-wrapper " + role;

    const avatar = document.createElement("div");
    avatar.className = "avatar " + (role === "user" ? "user-avatar" : "bot-avatar");
    avatar.innerHTML = role === "user" ? '<i class="fas fa-user"></i>' : '<i class="fas fa-robot"></i>';

    const content = document.createElement("div");
    content.className = "message-content";

    wrapper.appendChild(avatar);
    wrapper.appendChild(content);
    DOM.chatHistory.appendChild(wrapper);
    return content;
}

function scrollToBottom() {
    DOM.chatHistory.scrollTo({
        top: DOM.chatHistory.scrollHeight,
        behavior: "smooth",
    });
}

function clearChat() {
    document.querySelectorAll(".message-wrapper").forEach((el) => el.remove());
    DOM.emptyState.style.display = "block";
    DOM.chatInput.focus();
    conversationHistory = []; // Reset conversation memory
}

function buildEvidenceHtml(evTag) {
    if (!DOM.evidenceToggle.checked) return "";
    const evParts = evTag.replace("[EVIDENCE:", "").replace("]", "").trim();
    let evClass = "ok", evLabel = "Đủ bằng chứng";
    const lower = evParts.toLowerCase();
    if (lower.includes("mâu thuẫn") || lower.includes("conflict")) {
        evClass = "warning";
        evLabel = "Có mâu thuẫn";
    } else if (lower.includes("thiếu") || lower.includes("abstain")) {
        evClass = "error";
        evLabel = "Thiếu bằng chứng";
    }
    return `<div class="evidence-banner ${evClass}"><i class="fas fa-balance-scale"></i> ${escapeHtml(evLabel)}</div>`;
}

// ── API Key Management ────────────────────────────────────────────────

async function fetchApiKeys() {
    try {
        const res = await fetch(`${API_BASE_URL}/api/v1/admin/keys`);
        if (!res.ok) {
            // Panel may not exist if server hasn't been updated — hide it gracefully
            DOM.apiKeysPanel.style.display = "none";
            return;
        }
        const data = await res.json();
        DOM.apiKeysPanel.style.display = "";
        DOM.apiKeysCount.textContent = data.active || 0;
        renderApiKeys(data.keys || []);
    } catch {
        DOM.apiKeysPanel.style.display = "none";
    }
}

function renderApiKeys(keys) {
    DOM.apiKeysList.innerHTML = "";
    if (keys.length === 0) {
        DOM.apiKeysList.innerHTML = '<div class="api-key-item" style="cursor:default;color:var(--text-muted);font-size:0.75rem">Chưa có key nào</div>';
        return;
    }

    keys.forEach((k) => {
        const item = document.createElement("div");
        item.className = "api-key-item" + (k.revoked ? " revoked" : "");

        const dateStr = k.created_at ? new Date(k.created_at).toLocaleDateString("vi-VN") : "";

        item.innerHTML = `
            <div class="api-key-info">
                <span class="api-key-name">${escapeHtml(k.name)}</span>
                <span class="api-key-prefix">${escapeHtml(k.key_prefix)}</span>
                <span class="api-key-date">${dateStr}</span>
            </div>
            <span class="api-key-status ${k.revoked ? 'revoked' : 'active'}">
                ${k.revoked ? 'Đã thu hồi' : 'Hoạt động'}
            </span>
            ${!k.revoked ? `<button class="api-key-revoke" data-prefix="${escapeHtml(k.key_prefix)}" title="Thu hồi key">
                <i class="fas fa-ban"></i>
            </button>` : ''}
        `;

        // Revoke handler
        const revokeBtn = item.querySelector(".api-key-revoke");
        if (revokeBtn) {
            revokeBtn.addEventListener("click", (e) => {
                e.stopPropagation();
                onRevokeKey(k.key_prefix, k.name);
            });
        }

        DOM.apiKeysList.appendChild(item);
    });
}

function showCreateKeyForm() {
    DOM.createKeyForm.style.display = "block";
    DOM.btnCreateKey.style.display = "none";
    DOM.newKeyName.value = "";
    DOM.newKeyName.focus();
    DOM.apiKeysError.style.display = "none";
}

function cancelCreateKey() {
    DOM.createKeyForm.style.display = "none";
    DOM.btnCreateKey.style.display = "";
    DOM.newKeyName.value = "";
}

async function onCreateKey() {
    const name = DOM.newKeyName.value.trim();
    DOM.btnConfirmCreateKey.disabled = true;
    DOM.btnConfirmCreateKey.innerHTML = '<i class="fas fa-circle-notch fa-spin"></i> Đang tạo...';
    DOM.apiKeysError.style.display = "none";

    try {
        const res = await fetch(`${API_BASE_URL}/api/v1/admin/keys`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ name }),
        });

        if (!res.ok) {
            const err = await res.json().catch(() => ({ detail: res.statusText }));
            throw new Error(err.detail || "Unknown error");
        }

        const data = await res.json();

        // Hide form, show newly created key
        DOM.createKeyForm.style.display = "none";
        DOM.newKeyValue.textContent = data.key;
        DOM.newKeyDisplay.style.display = "block";

        // Refresh key list
        await fetchApiKeys();
    } catch (err) {
        DOM.apiKeysError.innerHTML = `<span style="color:var(--red)"><i class="fas fa-exclamation-circle"></i> ${escapeHtml(err.message)}</span>`;
        DOM.apiKeysError.style.display = "block";
    }

    DOM.btnConfirmCreateKey.disabled = false;
    DOM.btnConfirmCreateKey.innerHTML = '<i class="fas fa-plus"></i> Tạo';
    DOM.btnCreateKey.style.display = "";
}

function dismissNewKey() {
    DOM.newKeyDisplay.style.display = "none";
    DOM.newKeyValue.textContent = "";
}

function copyNewKey() {
    const key = DOM.newKeyValue.textContent;
    if (!key) return;

    if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(key).then(() => {
            showCopyFeedback(DOM.btnCopyKey);
        }).catch(() => {
            fallbackCopy(key, DOM.btnCopyKey);
        });
    } else {
        fallbackCopy(key, DOM.btnCopyKey);
    }
}

function fallbackCopy(text, btn) {
    const textarea = document.createElement("textarea");
    textarea.value = text;
    textarea.style.position = "fixed";
    textarea.style.opacity = "0";
    document.body.appendChild(textarea);
    textarea.select();
    try {
        document.execCommand("copy");
        showCopyFeedback(btn);
    } catch {}
    document.body.removeChild(textarea);
}

function showCopyFeedback(btn) {
    const icon = btn.querySelector("i");
    if (icon) {
        icon.className = "fas fa-check";
        btn.style.color = "var(--green)";
        setTimeout(() => {
            icon.className = "fas fa-copy";
            btn.style.color = "";
        }, 2000);
    }
}

async function onRevokeKey(prefix, name) {
    if (!confirm(`Thu hồi key "${name}" (${prefix})? Hành động này không thể hoàn tác.`)) {
        return;
    }

    try {
        const res = await fetch(`${API_BASE_URL}/api/v1/admin/keys/${encodeURIComponent(prefix)}`, {
            method: "DELETE",
        });

        if (!res.ok) {
            const err = await res.json().catch(() => ({ detail: res.statusText }));
            alert(`Lỗi: ${err.detail || "Không xác định"}`);
            return;
        }

        await fetchApiKeys();
    } catch (err) {
        alert(`Lỗi kết nối: ${err.message}`);
    }
}


function escapeHtml(str) {
    const div = document.createElement("div");
    div.textContent = str;
    return div.innerHTML;
}
