const config = window.CELL_AUTO_CONFIG;
const rows = config.rows;
const cols = config.cols;
const maxLiveCells = config.maxLiveCells;

const boardEditor = document.getElementById("board-editor");
const liveCount = document.getElementById("live-count");
const ruleInput = document.getElementById("rule-input");
const stepsInput = document.getElementById("steps-input");
const modelInput = document.getElementById("model-input");
const runStatus = document.getElementById("run-status");
const metricsOutput = document.getElementById("metrics-output");
const canvas = document.getElementById("simulation-canvas");
const frameSlider = document.getElementById("frame-slider");

const ctx = canvas.getContext("2d");
let board = createEmptyBoard();
let frames = [];
let animationHandle = null;

buildEditor();
loadBoardFromQuery();
renderBoardEditor();
drawFrame(board);

document.getElementById("clear-button").addEventListener("click", () => {
  board = createEmptyBoard();
  renderBoardEditor();
});

document.getElementById("random-button").addEventListener("click", () => {
  board = randomBoard();
  renderBoardEditor();
});

document.getElementById("manual-button").addEventListener("click", async () => {
  await runRequest("/api/evaluate/manual", {
    board,
    rule: ruleInput.value.trim(),
    max_steps: Number(stepsInput.value),
    max_live_fraction: 0.5,
    topology: "toroidal"
  });
});

document.getElementById("model-button").addEventListener("click", async () => {
  await runRequest("/api/evaluate/openrouter", {
    model: resolveModel(),
    rule: ruleInput.value.trim(),
    max_steps: Number(stepsInput.value),
    max_live_fraction: 0.5,
    topology: "toroidal"
  });
});

document.getElementById("play-button").addEventListener("click", playAnimation);
document.getElementById("pause-button").addEventListener("click", stopAnimation);

frameSlider.addEventListener("input", () => {
  const index = Number(frameSlider.value);
  if (frames[index]) {
    drawFrame(frames[index]);
  }
});

function createEmptyBoard() {
  return Array.from({ length: rows }, () => Array(cols).fill(0));
}

function loadBoardFromQuery() {
  const params = new URLSearchParams(window.location.search);
  const boardParam = params.get("board");
  if (!boardParam) {
    return;
  }

  try {
    const parsed = JSON.parse(boardParam);
    if (!Array.isArray(parsed) || parsed.length !== rows) {
      return;
    }
    const normalized = parsed.map((row) => {
      if (!Array.isArray(row) || row.length !== cols) {
        throw new Error("invalid board dimensions");
      }
      return row.map((value) => (value ? 1 : 0));
    });

    if (countLiveCells(normalized) > maxLiveCells) {
      return;
    }

    board = normalized;
    if (params.get("auto_run") === "1") {
      window.setTimeout(() => {
        document.getElementById("manual-button").click();
      }, 0);
    }
  } catch (error) {
    console.warn("[game-of-life-bench] failed to load board from query", error);
  }
}

function resolveModel() {
  const value = modelInput.value.trim();
  return value || config.defaultModel;
}

function randomBoard() {
  const next = createEmptyBoard();
  const positions = [];
  for (let r = 0; r < rows; r += 1) {
    for (let c = 0; c < cols; c += 1) {
      positions.push([r, c]);
    }
  }
  shuffle(positions);
  for (const [index, [r, c]] of positions.entries()) {
    if (index >= maxLiveCells) {
      break;
    }
    next[r][c] = Math.random() > 0.5 ? 1 : 0;
  }
  return next;
}

function shuffle(items) {
  for (let i = items.length - 1; i > 0; i -= 1) {
    const j = Math.floor(Math.random() * (i + 1));
    [items[i], items[j]] = [items[j], items[i]];
  }
}

function buildEditor() {
  boardEditor.style.gridTemplateColumns = `repeat(${cols}, 1fr)`;
  for (let r = 0; r < rows; r += 1) {
    for (let c = 0; c < cols; c += 1) {
      const cell = document.createElement("button");
      cell.type = "button";
      cell.className = "cell";
      cell.dataset.row = String(r);
      cell.dataset.col = String(c);
      cell.addEventListener("click", () => toggleCell(r, c));
      boardEditor.appendChild(cell);
    }
  }
}

function toggleCell(row, col) {
  board[row][col] = board[row][col] ? 0 : 1;
  if (countLiveCells(board) > maxLiveCells) {
    board[row][col] = 0;
    flashOverLimit();
    return;
  }
  renderBoardEditor();
}

function flashOverLimit() {
  boardEditor.querySelectorAll(".cell").forEach((cell) => {
    cell.classList.add("over-limit");
    setTimeout(() => cell.classList.remove("over-limit"), 220);
  });
}

function renderBoardEditor() {
  boardEditor.querySelectorAll(".cell").forEach((cell) => {
    const row = Number(cell.dataset.row);
    const col = Number(cell.dataset.col);
    cell.classList.toggle("alive", board[row][col] === 1);
  });
  const liveCells = countLiveCells(board);
  liveCount.textContent = `${liveCells} live cells`;
}

function countLiveCells(currentBoard) {
  return currentBoard.flat().reduce((sum, value) => sum + value, 0);
}

async function runRequest(endpoint, payload) {
  stopAnimation();
  runStatus.textContent = endpoint.includes("openrouter") ? "Waiting for model response..." : "Running simulation...";
  metricsOutput.textContent = endpoint.includes("openrouter")
    ? `Client POST -> ${endpoint}\nWaiting for server response...`
    : "Waiting for simulation results...";
  console.log("[game-of-life-bench] starting request", { endpoint, payload });
  try {
    const response = await fetch(endpoint, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });
    console.log("[game-of-life-bench] response received", { endpoint, status: response.status });
    const data = await response.json();
    if (!response.ok) {
      throw new Error(formatErrorDetail(data.detail));
    }

    frames = data.evaluation.simulation.frames;
    board = data.evaluation.simulation.initial_board;
    renderBoardEditor();
    frameSlider.max = String(Math.max(frames.length - 1, 0));
    frameSlider.value = "0";
    drawFrame(frames[0]);
    runStatus.textContent = `Run ${data.run_id} complete. Score ${data.evaluation.score}.`;
    metricsOutput.textContent = JSON.stringify(
      {
        score: data.evaluation.score,
        live_cells: data.evaluation.live_cells,
        repeat_step: data.evaluation.simulation.first_repeat_step,
        repeat_from_step: data.evaluation.simulation.repeat_from_step,
        period: data.evaluation.simulation.period,
        steps_completed: data.evaluation.simulation.steps_completed,
        populations: data.evaluation.simulation.populations
      },
      null,
      2
    );
    playAnimation();
  } catch (error) {
    console.error("[game-of-life-bench] request failed", { endpoint, error });
    runStatus.textContent = "Run failed.";
    metricsOutput.textContent = error.message;
  }
}

function formatErrorDetail(detail) {
  if (!detail) {
    return "Request failed.";
  }
  if (typeof detail === "string") {
    return detail;
  }

  const lines = [];
  if (detail.message) {
    lines.push(`error: ${detail.message}`);
  }
  if (detail.model) {
    lines.push(`model: ${detail.model}`);
  }
  if (detail.status_code) {
    lines.push(`status_code: ${detail.status_code}`);
  }
  if (detail.provider_error) {
    lines.push(`provider_error: ${detail.provider_error}`);
  }
  if (detail.raw_response) {
    lines.push("");
    lines.push("exact_response:");
    lines.push(String(detail.raw_response));
  }
  if (detail.prompt) {
    lines.push("");
    lines.push("prompt:");
    lines.push(String(detail.prompt));
  }
  return lines.join("\n") || "Request failed.";
}

function drawFrame(frame) {
  const size = Math.min(canvas.width / cols, canvas.height / rows);
  const offsetX = (canvas.width - cols * size) / 2;
  const offsetY = (canvas.height - rows * size) / 2;
  const deadFill = "#0c1311";
  const liveFill = "#8df7a9";
  const liveGlow = "#38c8ff";
  const gridLine = "rgba(126, 240, 163, 0.12)";
  const panelFill = "#050908";

  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = panelFill;
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.strokeStyle = gridLine;
  ctx.lineWidth = 1;

  for (let r = 0; r < rows; r += 1) {
    for (let c = 0; c < cols; c += 1) {
      const x = offsetX + c * size;
      const y = offsetY + r * size;
      ctx.fillStyle = deadFill;
      ctx.fillRect(x, y, size, size);
      if (frame[r][c]) {
        ctx.shadowColor = liveGlow;
        ctx.shadowBlur = 12;
        ctx.fillStyle = liveFill;
        ctx.fillRect(x + 4, y + 4, size - 8, size - 8);
        ctx.shadowBlur = 0;
      }
      ctx.strokeRect(x + 0.5, y + 0.5, size - 1, size - 1);
    }
  }
}

function playAnimation() {
  stopAnimation();
  if (!frames.length) {
    return;
  }
  let index = Number(frameSlider.value);
  animationHandle = window.setInterval(() => {
    drawFrame(frames[index]);
    frameSlider.value = String(index);
    index = (index + 1) % frames.length;
  }, 260);
}

function stopAnimation() {
  if (animationHandle !== null) {
    window.clearInterval(animationHandle);
    animationHandle = null;
  }
}
