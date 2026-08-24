// AutoSE-RAG v2 Client Logic

let activeHitlSessionId = null;

document.addEventListener("DOMContentLoaded", () => {
    initNavigation();
    initPresets();
    initPipelineRunner();
    initBenchmarkRunner();
    initHitlModal();
    pollHitlSessions();
});

// Navigation Tabs
function initNavigation() {
    const navButtons = document.querySelectorAll(".nav-item");
    const tabContents = document.querySelectorAll(".tab-content");
    const pageTitle = document.getElementById("page-title");

    const titles = {
        "pipeline-tab": "Autonomous Agent Pipeline",
        "hitl-tab": "Human-in-the-Loop Review Queue",
        "trace-tab": "Bi-Directional Traceability Matrix",
        "eval-tab": "Comparative Research Benchmark Lab"
    };

    navButtons.forEach(btn => {
        btn.addEventListener("click", () => {
            const target = btn.getAttribute("data-tab");
            navButtons.forEach(b => b.classList.remove("active"));
            tabContents.forEach(t => t.classList.remove("active"));

            btn.classList.add("active");
            document.getElementById(target).classList.add("active");
            pageTitle.innerText = titles[target] || "Agentic Software Engineering";

            if (target === "hitl-tab") {
                loadHitlQueue();
            }
        });
    });
}

// Preset Requirement Selector
function initPresets() {
    const presetDropdown = document.getElementById("preset-dropdown");
    const reqInput = document.getElementById("req-input");

    const presets = {
        "pps": "The system shall validate dual redundant accelerator pedal position sensors (PPS1, PPS2) between 0.5V and 4.5V and reject readings if discrepancy exceeds 0.2V.",
        "e2e": "The system shall verify AUTOSAR E2E message frames by calculating CRC-32 over Data ID, alive counter, and payload, discarding frames with CRC mismatch or sequence gap.",
        "dtc": "The system shall debounce coolant temperature sensor faults for 3 consecutive execution cycles before confirming Diagnostic Trouble Code DTC_COOLANT_TEMP_FAULT and storing freeze frame data.",
        "watchdog": "The system shall maintain a watchdog heartbeat timer and transition to safe state (torque disable) if heartbeat is not refreshed within 50ms Fault Tolerant Time Interval."
    };

    presetDropdown.addEventListener("change", (e) => {
        const val = e.target.value;
        if (presets[val]) {
            reqInput.value = presets[val];
        }
    });
}

// Agent Pipeline Execution
function initPipelineRunner() {
    const runBtn = document.getElementById("run-pipeline-btn");
    const reqInput = document.getElementById("req-input");
    const modeSelect = document.getElementById("retrieval-mode");
    const hitlToggle = document.getElementById("hitl-toggle");

    runBtn.addEventListener("click", async () => {
        const requirement = reqInput.value.trim();
        if (!requirement) return alert("Please enter a software requirement specification.");

        runBtn.disabled = true;
        runBtn.innerHTML = `Running Agent Pipeline...`;
        resetStageVisualizers();

        try {
            // Animate stage 1
            setStageActive("stage-retrieval");
            await delay(400);

            const resp = await fetch("/api/agent/pipeline/run", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    requirement: requirement,
                    mode: modeSelect.value,
                    top_k: 4,
                    require_human_approval: hitlToggle.checked,
                    max_revisions: 2
                })
            });

            const data = await resp.json();
            setStageCompleted("stage-retrieval");
            setStageCompleted("stage-codegen");
            setStageCompleted("stage-testgen");

            if (data.status === "PENDING_APPROVAL") {
                alert(`Human approval required! Session ${data.session_id} added to HITL Queue.`);
                openHitlModal(data.session_id, data.code, data.tests);
                pollHitlSessions();
                return;
            }

            setStageCompleted("stage-sandbox");
            setStageCompleted("stage-critic");
            setStageCompleted("stage-trace");

            renderPipelineResults(data);
            renderTraceability(data.traceability);

        } catch (err) {
            console.error("Pipeline error:", err);
            alert("Error executing pipeline: " + err.message);
        } finally {
            runBtn.disabled = false;
            runBtn.innerHTML = `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="5 3 19 12 5 21 5 3"></polygon></svg> Execute Agent Pipeline`;
        }
    });
}

function renderPipelineResults(data) {
    document.getElementById("pipeline-results").style.display = "grid";

    // Evidence
    const evList = document.getElementById("evidence-list");
    evList.innerHTML = "";
    document.getElementById("evidence-badge").innerText = `${data.evidence.length} Chunks`;

    data.evidence.forEach(e => {
        const div = document.createElement("div");
        div.className = "evidence-card";
        div.innerHTML = `
            <div class="evidence-head">
                <span>[${e.id}] ${e.source}</span>
                <span>Score: ${e.score}</span>
            </div>
            <div class="evidence-body">${e.text}</div>
        `;
        evList.appendChild(div);
    });

    // Code & Tests
    document.getElementById("code-block").innerText = data.code;
    document.getElementById("tests-block").innerText = data.tests;

    // Sandbox output
    const execRes = data.execution_result || {};
    const outBox = document.getElementById("sandbox-output");
    outBox.innerText = execRes.stdout || execRes.stderr || "Tests executed successfully.";

    const badge = document.getElementById("exec-status-badge");
    if (execRes.success) {
        badge.className = "badge badge-success";
        badge.innerText = `All Passed (${execRes.passed_count}/${execRes.total_tests})`;
    } else {
        badge.className = "badge badge-danger";
        badge.innerText = `Failures (${execRes.failed_count}/${execRes.total_tests})`;
    }

    // Critic box
    const criticBox = document.getElementById("critic-verdict-box");
    const critic = data.critic_review || {};
    criticBox.innerHTML = `<strong>Critic Agent Verdict:</strong> [${critic.verdict || "ACCEPT"}] ${critic.critique || "Verified safety compliance."}`;
}

function renderTraceability(trace) {
    if (!trace) return;
    document.getElementById("trace-score-val").innerText = trace.completeness_score.toFixed(2);
    const tbody = document.getElementById("traceability-body");
    tbody.innerHTML = "";

    if (trace.matrix_mappings && trace.matrix_mappings.length > 0) {
        trace.matrix_mappings.forEach((m, idx) => {
            const evRef = trace.evidence_references[idx % trace.evidence_references.length] || "N/A";
            const tr = document.createElement("tr");
            tr.innerHTML = `
                <td><code>REQ-${idx + 1}</code></td>
                <td><span class="badge" style="background: rgba(59,130,246,0.15); color: #60a5fa;">${evRef}</span></td>
                <td><code>${m.target_symbol}()</code></td>
                <td><code>${m.test_case}</code></td>
                <td><span class="badge ${m.verified ? 'badge-success' : 'badge-danger'}">${m.verified ? 'VERIFIED' : 'FAILED'}</span></td>
            `;
            tbody.appendChild(tr);
        });
    }
}

// HITL Logic
async function pollHitlSessions() {
    try {
        const res = await fetch("/api/hitl/sessions");
        const json = await res.json();
        const pending = (json.sessions || []).filter(s => s.status === "PENDING_APPROVAL");
        document.getElementById("hitl-count-badge").innerText = pending.length;
    } catch (e) {}
}

async function loadHitlQueue() {
    const list = document.getElementById("hitl-queue-list");
    try {
        const res = await fetch("/api/hitl/sessions");
        const json = await res.json();
        const sessions = json.sessions || [];

        if (sessions.length === 0) {
            list.innerHTML = `<div class="empty-state">No pending reviews. Enable "Require Human Approval" in the Pipeline tab to test HITL workflows.</div>`;
            return;
        }

        list.innerHTML = "";
        sessions.forEach(sess => {
            const div = document.createElement("div");
            div.className = "card glass-card";
            div.style.marginBottom = "14px";
            div.innerHTML = `
                <div class="flex-between">
                    <div>
                        <h4>Session ${sess.session_id}</h4>
                        <p class="text-muted">${sess.data.requirement}</p>
                    </div>
                    <div>
                        <span class="badge ${sess.status === 'PENDING_APPROVAL' ? 'badge-danger' : 'badge-success'}">${sess.status}</span>
                        ${sess.status === 'PENDING_APPROVAL' ? `<button class="btn btn-primary" style="margin-left: 10px;" onclick="openHitlModal('${sess.session_id}', \`${escapeStr(sess.data.code)}\`, \`${escapeStr(sess.data.tests)}\`)">Review Artifacts</button>` : ''}
                    </div>
                </div>
            `;
            list.appendChild(div);
        });
    } catch (e) {
        console.error(e);
    }
}

function initHitlModal() {
    document.getElementById("modal-approve-btn").addEventListener("click", async () => {
        if (!activeHitlSessionId) return;
        const res = await fetch(`/api/hitl/sessions/${activeHitlSessionId}/approve`, { method: "POST" });
        const updated = await res.json();
        closeHitlModal();
        alert("Session approved and verified in Sandbox!");
        loadHitlQueue();
        pollHitlSessions();
        renderPipelineResults(updated.data);
        renderTraceability(updated.data.traceability);
    });

    document.getElementById("modal-reject-btn").addEventListener("click", async () => {
        if (!activeHitlSessionId) return;
        const feedback = document.getElementById("modal-feedback-input").value;
        const res = await fetch(`/api/hitl/sessions/${activeHitlSessionId}/reject`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ feedback: feedback })
        });
        const updated = await res.json();
        closeHitlModal();
        alert("Rejection submitted. Agent refined implementation based on feedback.");
        loadHitlQueue();
        pollHitlSessions();
    });
}

function openHitlModal(sessionId, code, tests) {
    activeHitlSessionId = sessionId;
    document.getElementById("modal-code-view").innerText = code;
    document.getElementById("modal-tests-view").innerText = tests;
    document.getElementById("modal-feedback-input").value = "";
    document.getElementById("hitl-modal").style.display = "flex";
}

function closeHitlModal() {
    document.getElementById("hitl-modal").style.display = "none";
    activeHitlSessionId = null;
}

// Benchmark Suite Runner (Exp 1 & Exp 2)
function initBenchmarkRunner() {
    const runRetBtn = document.getElementById("run-ret-btn");
    const runGenBtn = document.getElementById("run-gen-btn");
    const spinner = document.getElementById("benchmark-spinner");
    const spinnerMsg = document.getElementById("spinner-msg");

    // Experiment 1 (Retrieval)
    if (runRetBtn) {
        runRetBtn.addEventListener("click", async () => {
            runRetBtn.disabled = true;
            spinnerMsg.innerText = "Executing Experiment 1: Evaluating TF-IDF vs FAISS vs Hybrid RRF...";
            spinner.style.display = "flex";

            try {
                const resp = await fetch("/api/eval/run_retrieval_benchmark?limit=15", { method: "POST" });
                const data = await resp.json();

                renderRetrievalTable(data.summary);
                document.getElementById("latex-ret-block").innerText = data.latex_table;
                document.getElementById("retrieval-chart-container").innerHTML = `<img src="/static/retrieval_comparison.png?t=${Date.now()}" style="max-width: 100%; border-radius: 8px; border: 1px solid var(--surface-glass-border);" alt="Retrieval Chart" />`;
            } catch (err) {
                alert("Error running retrieval benchmark: " + err.message);
            } finally {
                spinner.style.display = "none";
                runRetBtn.disabled = false;
            }
        });
    }

    // Experiment 2 (Generation & Hidden Oracles)
    if (runGenBtn) {
        runGenBtn.addEventListener("click", async () => {
            runGenBtn.disabled = true;
            spinnerMsg.innerText = "Executing Experiment 2: Evaluating M0 vs M1 vs M2 against Hidden Oracle Tests...";
            spinner.style.display = "flex";

            try {
                const resp = await fetch("/api/eval/run_generation_benchmark?limit=15", { method: "POST" });
                const data = await resp.json();

                renderGenerationTable(data.summary);
                document.getElementById("latex-gen-block").innerText = data.latex_table;
                document.getElementById("generation-chart-container").innerHTML = `<img src="/static/generation_comparison.png?t=${Date.now()}" style="max-width: 100%; border-radius: 8px; border: 1px solid var(--surface-glass-border);" alt="Generation Chart" />`;
            } catch (err) {
                alert("Error running generation benchmark: " + err.message);
            } finally {
                spinner.style.display = "none";
                runGenBtn.disabled = false;
            }
        });
    }

    // Initial render of default tables
    fetch("/api/eval/run_retrieval_benchmark?limit=15", { method: "POST" })
        .then(r => r.json())
        .then(d => renderRetrievalTable(d.summary))
        .catch(() => {});

    fetch("/api/eval/run_generation_benchmark?limit=15", { method: "POST" })
        .then(r => r.json())
        .then(d => renderGenerationTable(d.summary))
        .catch(() => {});
}

function renderRetrievalTable(summary) {
    const tableDiv = document.getElementById("retrieval-metrics-table");
    if (!summary || !tableDiv) return;

    let html = `<table class="data-table"><thead><tr><th>IR Metric</th><th>Sparse (TF-IDF)</th><th>Dense (FAISS)</th><th>Hybrid RRF (Ours)</th></tr></thead><tbody>`;
    const metrics = ["Precision@1", "Precision@3", "Precision@4", "Recall@4", "MRR", "NDCG@4"];
    metrics.forEach(m => {
        html += `<tr>
            <td><strong>${m}</strong></td>
            <td>${summary.tfidf_sparse[m]}</td>
            <td>${summary.faiss_dense[m]}</td>
            <td style="color: #34d399; font-weight: 700;">${summary.hybrid_rrf[m]}</td>
        </tr>`;
    });
    html += `</tbody></table>`;
    tableDiv.innerHTML = html;
}

function renderGenerationTable(summary) {
    const tableDiv = document.getElementById("generation-metrics-table");
    if (!summary || !tableDiv) return;

    let html = `<table class="data-table"><thead><tr><th>Software Metric</th><th>M0: Raw LLM</th><th>M1: Standard RAG</th><th>M2: Agentic RAG (Ours)</th></tr></thead><tbody>`;
    const keys = Object.keys(summary.no_rag || {});
    keys.forEach(k => {
        html += `<tr>
            <td><strong>${k}</strong></td>
            <td>${summary.no_rag[k]}</td>
            <td>${summary.standard_rag[k]}</td>
            <td style="color: #34d399; font-weight: 700;">${summary.agentic_rag[k]}</td>
        </tr>`;
    });
    html += `</tbody></table>`;
    tableDiv.innerHTML = html;
}

// Visual Stage Helpers
function setStageActive(id) {
    const node = document.getElementById(id);
    if (node) node.className = "stage-node active";
}

function setStageCompleted(id) {
    const node = document.getElementById(id);
    if (node) node.className = "stage-node completed";
}

function resetStageVisualizers() {
    ["stage-retrieval", "stage-codegen", "stage-testgen", "stage-sandbox", "stage-critic", "stage-trace"].forEach(id => {
        const node = document.getElementById(id);
        if (node) node.className = "stage-node";
    });
}

function copyCode(elementId) {
    const text = document.getElementById(elementId).innerText;
    navigator.clipboard.writeText(text);
    alert("Copied to clipboard!");
}

function delay(ms) {
    return new Promise(res => setTimeout(res, ms));
}

function escapeStr(str) {
    if (!str) return "";
    return str.replace(/`/g, "\\`").replace(/\$/g, "\\$");
}
