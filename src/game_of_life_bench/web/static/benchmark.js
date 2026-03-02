const benchmarkConfig = window.CELL_AUTO_BENCHMARK_CONFIG;
const benchmarkLog = document.getElementById("benchmark-log");
const leaderboardTable = document.getElementById("leaderboard-table");
const benchmarkViewCanvas = document.getElementById("benchmark-view-canvas");
const bestBoardModel = document.getElementById("best-board-model");
const benchmarkViewLabel = document.getElementById("benchmark-view-label");
const simulateBoardButton = document.getElementById("simulate-board-button");
const benchmarkSimStatus = document.getElementById("benchmark-sim-status");
const benchmarkPlayButton = document.getElementById("benchmark-play-button");
const benchmarkPauseButton = document.getElementById("benchmark-pause-button");
const benchmarkFrameSlider = document.getElementById("benchmark-frame-slider");
const benchmarkCtx = benchmarkViewCanvas.getContext("2d");
const sortState = { key: "submission_score", direction: "desc" };

let selectedEntry = null;
let simFrames = [];
let simAnimationHandle = null;

if (simulateBoardButton) {
  simulateBoardButton.addEventListener("click", simulateSelectedBoard);
}

if (benchmarkPlayButton) {
  benchmarkPlayButton.addEventListener("click", playSimulation);
}

if (benchmarkPauseButton) {
  benchmarkPauseButton.addEventListener("click", stopSimulation);
}

if (benchmarkFrameSlider) {
  benchmarkFrameSlider.addEventListener("input", () => {
    const index = Number(benchmarkFrameSlider.value);
    if (simFrames[index]) {
      drawBoard(simFrames[index]);
    }
  });
}

bindStaticLeaderboard();

function bindStaticLeaderboard() {
  const table = leaderboardTable.querySelector("table");
  if (!table) {
    return;
  }
  const rows = Array.from(table.querySelectorAll("tbody tr"));
  if (!rows.length) {
    return;
  }
  rows.forEach((row, index) => {
    const entry = {
      model: row.dataset.model,
      best_board: JSON.parse(row.dataset.bestBoard),
      submission_score: Number(row.dataset.submissionScore),
      best_average_score: Number(row.dataset.bestAverageScore),
      avg_output_tokens: row.dataset.avgOutputTokens ? Number(row.dataset.avgOutputTokens) : null,
      best_run_id: row.dataset.bestRunId,
      best_benchmark_id: row.dataset.bestBenchmarkId,
      total_cost: row.dataset.totalCost ? Number(row.dataset.totalCost) : null
    };
    row.addEventListener("click", () => renderBestBoard(entry));
    if (index === 0) {
      row.classList.add("selected");
      renderBestBoard(entry);
    }
  });
  decorateSortHeaders(table);
  bindSortHeaders(table);
}

function renderBestBoard(entry) {
  selectedEntry = entry;
  stopSimulation();
  simFrames = [];
  benchmarkFrameSlider.max = "0";
  benchmarkFrameSlider.value = "0";
  bestBoardModel.textContent = entry.model;
  benchmarkViewLabel.textContent = "submission";
  benchmarkSimStatus.textContent = "idle";
  benchmarkLog.textContent = "Select a model and load its simulation.";
  drawBoard(entry.best_board);

  document.querySelectorAll("#leaderboard-table tbody tr").forEach((row) => row.classList.remove("selected"));
  const rows = Array.from(document.querySelectorAll("#leaderboard-table tbody tr"));
  const selectedRow = rows.find((row) => row.dataset.model === entry.model);
  if (selectedRow) {
    selectedRow.classList.add("selected");
  }
}

async function simulateSelectedBoard() {
  if (!selectedEntry) {
    return;
  }
  stopSimulation();
  benchmarkViewLabel.textContent = "simulation";
  benchmarkSimStatus.textContent = "loading...";
  try {
    const response = await fetch("/api/evaluate/manual", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        board: selectedEntry.best_board,
        rule: "B3/S23",
        max_steps: 1000,
        max_live_fraction: 0.5,
        topology: "toroidal"
      })
    });
    const data = await response.json();
    if (!response.ok) {
      throw new Error(typeof data.detail === "string" ? data.detail : JSON.stringify(data.detail, null, 2));
    }
    simFrames = data.evaluation.simulation.frames;
    benchmarkFrameSlider.max = String(Math.max(simFrames.length - 1, 0));
    benchmarkFrameSlider.value = "0";
    drawBoard(simFrames[0]);
    benchmarkSimStatus.textContent = `score ${data.evaluation.score}`;
    benchmarkLog.textContent = `Loaded ${selectedEntry.model} submission. Score ${data.evaluation.score}.`;
    playSimulation();
  } catch (error) {
    benchmarkSimStatus.textContent = "simulation failed";
    benchmarkLog.textContent = error.message;
  }
}

function playSimulation() {
  stopSimulation();
  if (!simFrames.length) {
    benchmarkLog.textContent = "Load a simulation first.";
    return;
  }
  benchmarkSimStatus.textContent = "playing";
  let index = Number(benchmarkFrameSlider.value);
  simAnimationHandle = window.setInterval(() => {
    drawBoard(simFrames[index]);
    benchmarkFrameSlider.value = String(index);
    index = (index + 1) % simFrames.length;
  }, 260);
}

function stopSimulation() {
  if (simAnimationHandle !== null) {
    window.clearInterval(simAnimationHandle);
    simAnimationHandle = null;
    if (simFrames.length) {
      benchmarkSimStatus.textContent = "paused";
    }
  }
}

function drawBoard(board) {
  drawGridBoard(benchmarkCtx, benchmarkViewCanvas, board);
}

function drawGridBoard(context, canvas, board) {
  const rows = benchmarkConfig.rows;
  const cols = benchmarkConfig.cols;
  const size = Math.min(canvas.width / cols, canvas.height / rows);
  const deadFill = "#0c1311";
  const liveFill = "#8df7a9";
  const gridLine = "rgba(126, 240, 163, 0.12)";

  context.clearRect(0, 0, canvas.width, canvas.height);
  context.fillStyle = "#050908";
  context.fillRect(0, 0, canvas.width, canvas.height);
  context.strokeStyle = gridLine;
  context.lineWidth = 1;

  for (let r = 0; r < rows; r += 1) {
    for (let c = 0; c < cols; c += 1) {
      const x = c * size;
      const y = r * size;
      context.fillStyle = deadFill;
      context.fillRect(x, y, size, size);
      if (board && board[r] && board[r][c]) {
        context.fillStyle = liveFill;
        context.fillRect(x + 3, y + 3, size - 6, size - 6);
      }
      context.strokeRect(x + 0.5, y + 0.5, size - 1, size - 1);
    }
  }
}

function formatCost(value) {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return "-";
  }
  return `$${Number(value).toFixed(4)}`;
}

function formatAvgOutputTokens(value) {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return "-";
  }
  return Number(value).toFixed(1);
}

function splitModel(modelId) {
  const slashIndex = modelId.indexOf("/");
  if (slashIndex === -1) {
    return { provider: "unknown", name: modelId };
  }
  return {
    provider: modelId.slice(0, slashIndex),
    name: modelId.slice(slashIndex + 1)
  };
}

function bindSortHeaders(table) {
  table.querySelectorAll("th[data-sort-key]").forEach((header) => {
    header.addEventListener("click", () => {
      const key = header.dataset.sortKey;
      if (sortState.key === key) {
        sortState.direction = sortState.direction === "asc" ? "desc" : "asc";
      } else {
        sortState.key = key;
        sortState.direction = key === "provider" || key === "model_name" ? "asc" : "desc";
      }
      sortStaticTable(table);
      decorateSortHeaders(table);
    });
  });
}

function decorateSortHeaders(table) {
  table.querySelectorAll("th[data-sort-key]").forEach((header) => {
    const key = header.dataset.sortKey;
    const base = header.textContent.replace(/[▲▼]$/, "").trim();
    if (key === sortState.key) {
      header.textContent = `${base}${sortState.direction === "asc" ? " ▲" : " ▼"}`;
    } else {
      header.textContent = base;
    }
  });
}

function sortStaticTable(table) {
  const tbody = table.querySelector("tbody");
  const rows = Array.from(tbody.querySelectorAll("tr"));
  rows.sort((a, b) => compareRowData(rowToEntry(a), rowToEntry(b)));
  rows.forEach((row, index) => {
    row.children[0].textContent = String(index + 1);
    tbody.appendChild(row);
  });
}

function rowToEntry(row) {
  return {
    rank: Number(row.dataset.rank),
    provider: row.dataset.provider,
    model_name: row.dataset.modelName,
    submission_score: Number(row.dataset.submissionScore),
    best_average_score: Number(row.dataset.bestAverageScore),
    avg_output_tokens: row.dataset.avgOutputTokens ? Number(row.dataset.avgOutputTokens) : null,
    total_cost: row.dataset.totalCost ? Number(row.dataset.totalCost) : null
  };
}

function compareRowData(a, b) {
  const direction = sortState.direction === "asc" ? 1 : -1;
  const key = sortState.key;
  const aValue = a[key];
  const bValue = b[key];

  if (typeof aValue === "string" || typeof bValue === "string") {
    return String(aValue).localeCompare(String(bValue)) * direction;
  }

  const aNumber = aValue == null ? Number.NEGATIVE_INFINITY : Number(aValue);
  const bNumber = bValue == null ? Number.NEGATIVE_INFINITY : Number(bValue);
  if (aNumber === bNumber) {
    return 0;
  }
  return (aNumber - bNumber) * direction;
}
