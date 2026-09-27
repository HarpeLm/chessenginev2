"use strict";

/* ═══════════════════════════ Outils ═══════════════════════════ */

const $ = (selector, root = document) => root.querySelector(selector);
const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];
const FILES = "abcdefgh";
const MINUS = "−";

const squareName = (sq) => FILES[sq % 8] + (Math.floor(sq / 8) + 1);
const squareIndex = (name) => FILES.indexOf(name[0]) + (name.charCodeAt(1) - 49) * 8;

/** FEN → tableau de 64 cases (a1 = 0) contenant « wN », « bP »… ou null. */
function parseFen(fen) {
  const board = new Array(64).fill(null);
  const rows = fen.split(" ")[0].split("/");
  rows.forEach((row, i) => {
    const rank = 7 - i;
    let file = 0;
    for (const ch of row) {
      if (/\d/.test(ch)) {
        file += Number(ch);
      } else {
        const color = ch === ch.toUpperCase() ? "w" : "b";
        board[rank * 8 + file] = color + ch.toUpperCase();
        file += 1;
      }
    }
  });
  return board;
}

async function api(path, body) {
  const options = body === undefined
    ? {}
    : { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body) };
  const response = await fetch(path, options);
  const data = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(data.error || response.statusText);
  return data;
}

let toastTimer = null;
function toast(message) {
  const el = $("#toast");
  el.textContent = message;
  el.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { el.hidden = true; }, 3200);
}

const fmtInt = (n) => Math.round(n).toLocaleString("fr-FR");
const fmtDecimal = (n, digits) => n.toFixed(digits).replace(".", ",");
const fmtSigned = (n) => (n > 0 ? "+" : n < 0 ? MINUS : "") + fmtInt(Math.abs(n));

function fmtCompact(n) {
  if (n >= 1e6) return fmtDecimal(n / 1e6, 1) + " M";
  if (n >= 1e4) return fmtInt(n / 1e3) + " k";
  return fmtInt(n);
}

/** Évaluation du point de vue des Blancs : « +0,35 », « M3 », « −M2 ». */
function fmtEval(info) {
  if (!info) return "0,00";
  if (info.mate !== null && info.mate !== undefined) {
    return (info.mate > 0 ? "M" : MINUS + "M") + Math.abs(info.mate);
  }
  const pawns = info.cp / 100;
  const sign = pawns > 0.004 ? "+" : pawns < -0.004 ? MINUS : "";
  return sign + fmtDecimal(Math.abs(pawns), 2);
}

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

/* ═══════════════════════════ Échiquier ═══════════════════════════ */

class BoardView {
  constructor(root, options = {}) {
    this.root = root;
    this.options = options;
    this.orientation = "w";
    this.position = new Array(64).fill(null);
    this.pieceEls = new Map();
    this.legal = [];
    this.selected = null;
    this.lastMove = null;
    this.check = null;
    this.drag = null;

    root.innerHTML = "";
    this.squaresEl = el("div", "squares");
    this.piecesEl = el("div", "pieces");
    root.append(this.squaresEl, this.piecesEl);
    this.buildSquares();

    if (options.interactive) {
      root.classList.add("is-interactive");
      root.addEventListener("pointerdown", (e) => this.onDown(e));
      window.addEventListener("pointermove", (e) => this.onMove(e));
      window.addEventListener("pointerup", (e) => this.onUp(e));
      window.addEventListener("pointercancel", () => this.cancelDrag());
    }
  }

  squareAt(col, row) {
    return this.orientation === "w" ? (7 - row) * 8 + col : row * 8 + (7 - col);
  }

  colRow(sq) {
    const file = sq % 8;
    const rank = Math.floor(sq / 8);
    return this.orientation === "w" ? [file, 7 - rank] : [7 - file, rank];
  }

  buildSquares() {
    this.squaresEl.innerHTML = "";
    this.sqEls = new Array(64);
    for (let row = 0; row < 8; row++) {
      for (let col = 0; col < 8; col++) {
        const sq = this.squareAt(col, row);
        const file = sq % 8;
        const rank = Math.floor(sq / 8);
        const cell = el("div", (file + rank) % 2 === 0 ? "sq is-dark" : "sq");
        if (col === 0) cell.append(el("span", "coord coord-rank", String(rank + 1)));
        if (row === 7) cell.append(el("span", "coord coord-file", FILES[file]));
        this.sqEls[sq] = cell;
        this.squaresEl.append(cell);
      }
    }
  }

  place(pieceEl, sq, animate = true) {
    const [col, row] = this.colRow(sq);
    if (!animate) pieceEl.style.transition = "none";
    pieceEl.style.transform = `translate(${col * 100}%, ${row * 100}%)`;
    if (!animate) {
      void pieceEl.offsetWidth;
      pieceEl.style.transition = "";
    }
  }

  setOrientation(orientation) {
    if (orientation === this.orientation) return;
    this.orientation = orientation;
    this.buildSquares();
    for (const [sq, pieceEl] of this.pieceEls) this.place(pieceEl, sq, false);
    this.renderMarks();
  }

  createPiece(code) {
    const pieceEl = el("div", "piece");
    pieceEl.dataset.piece = code;
    pieceEl.style.backgroundImage = `url(/pieces/${code}.svg)`;
    return pieceEl;
  }

  /** Affiche une position ; les pièces qui ont bougé glissent jusqu'à leur nouvelle case. */
  setPosition(fen, { lastMove = null, check = null } = {}) {
    const next = parseFen(fen);
    const kept = new Map();
    const pool = [];
    for (const [sq, pieceEl] of this.pieceEls) {
      if (next[sq] === pieceEl.dataset.piece) kept.set(sq, pieceEl);
      else pool.push([sq, pieceEl]);
    }
    const distance = (a, b) => Math.abs((a % 8) - (b % 8)) + Math.abs(Math.floor(a / 8) - Math.floor(b / 8));
    for (let sq = 0; sq < 64; sq++) {
      const code = next[sq];
      if (!code || kept.has(sq)) continue;
      let best = -1;
      let bestDistance = Infinity;
      pool.forEach(([from, pieceEl], i) => {
        if (pieceEl.dataset.piece === code && distance(from, sq) < bestDistance) {
          best = i;
          bestDistance = distance(from, sq);
        }
      });
      if (best >= 0) {
        const [, pieceEl] = pool.splice(best, 1)[0];
        this.place(pieceEl, sq);
        kept.set(sq, pieceEl);
      } else {
        const pieceEl = this.createPiece(code);
        pieceEl.classList.add("is-entering");
        this.place(pieceEl, sq, false);
        this.piecesEl.append(pieceEl);
        void pieceEl.offsetWidth;
        pieceEl.classList.remove("is-entering");
        kept.set(sq, pieceEl);
      }
    }
    for (const [, pieceEl] of pool) {
      pieceEl.classList.add("is-leaving");
      setTimeout(() => pieceEl.remove(), 240);
    }
    this.pieceEls = kept;
    this.position = next;
    this.lastMove = lastMove;
    this.check = check;
    this.selected = null;
    this.renderMarks();
  }

  setLegal(ucis) {
    this.legal = ucis.map((uci) => ({
      uci,
      from: squareIndex(uci.slice(0, 2)),
      to: squareIndex(uci.slice(2, 4)),
      promotion: uci[4] || null,
    }));
    if (this.selected !== null && !this.movesFrom(this.selected).length) this.selected = null;
    this.renderMarks();
  }

  movesFrom(sq) {
    return this.legal.filter((m) => m.from === sq);
  }

  renderMarks() {
    for (const cell of this.sqEls) {
      cell.classList.remove("is-last", "is-selected", "is-check", "is-target", "is-occupied", "is-hover");
    }
    if (this.lastMove) {
      for (const name of this.lastMove) this.sqEls[squareIndex(name)].classList.add("is-last");
    }
    if (this.check) this.sqEls[squareIndex(this.check)].classList.add("is-check");
    if (this.selected !== null) {
      this.sqEls[this.selected].classList.add("is-selected");
      for (const move of this.movesFrom(this.selected)) {
        const cell = this.sqEls[move.to];
        cell.classList.add("is-target");
        if (this.position[move.to]) cell.classList.add("is-occupied");
      }
    }
  }

  squareFromEvent(e) {
    const rect = this.root.getBoundingClientRect();
    const col = Math.floor(((e.clientX - rect.left) / rect.width) * 8);
    const row = Math.floor(((e.clientY - rect.top) / rect.height) * 8);
    if (col < 0 || col > 7 || row < 0 || row > 7) return null;
    return this.squareAt(col, row);
  }

  onDown(e) {
    if (e.button !== 0) return;
    const sq = this.squareFromEvent(e);
    if (sq === null) return;
    if (this.selected !== null && this.movesFrom(this.selected).some((m) => m.to === sq)) {
      this.tryMove(this.selected, sq);
      return;
    }
    const pieceEl = this.pieceEls.get(sq);
    if (pieceEl && this.movesFrom(sq).length) {
      this.drag = { sq, pieceEl, x: e.clientX, y: e.clientY, moved: false, wasSelected: this.selected === sq };
      this.selected = sq;
      this.renderMarks();
      e.preventDefault();
    } else if (this.selected !== null) {
      this.selected = null;
      this.renderMarks();
    }
  }

  onMove(e) {
    const drag = this.drag;
    if (!drag) return;
    if (!drag.moved && Math.hypot(e.clientX - drag.x, e.clientY - drag.y) < 4) return;
    drag.moved = true;
    const rect = this.root.getBoundingClientRect();
    const size = rect.width / 8;
    drag.pieceEl.classList.add("is-dragging");
    drag.pieceEl.style.transform =
      `translate(${e.clientX - rect.left - size / 2}px, ${e.clientY - rect.top - size / 2}px)`;
    const over = this.squareFromEvent(e);
    for (const cell of this.sqEls) cell.classList.remove("is-hover");
    if (over !== null && this.movesFrom(drag.sq).some((m) => m.to === over)) {
      this.sqEls[over].classList.add("is-hover");
    }
  }

  onUp(e) {
    const drag = this.drag;
    if (!drag) return;
    this.drag = null;
    for (const cell of this.sqEls) cell.classList.remove("is-hover");
    if (!drag.moved) {
      // Second clic sur la pièce déjà sélectionnée : on la repose.
      if (drag.wasSelected) {
        this.selected = null;
        this.renderMarks();
      }
      return;
    }
    drag.pieceEl.classList.remove("is-dragging");
    const target = this.squareFromEvent(e);
    if (target !== null && target !== drag.sq && this.movesFrom(drag.sq).some((m) => m.to === target)) {
      this.place(drag.pieceEl, target, false);
      this.tryMove(drag.sq, target);
    } else {
      this.place(drag.pieceEl, drag.sq);
    }
  }

  cancelDrag() {
    if (!this.drag) return;
    this.drag.pieceEl.classList.remove("is-dragging");
    this.place(this.drag.pieceEl, this.drag.sq);
    this.drag = null;
  }

  async tryMove(from, to) {
    const candidates = this.legal.filter((m) => m.from === from && m.to === to);
    if (!candidates.length) return;
    let uci = candidates[0].uci;
    if (candidates.length > 1) {
      const color = this.position[from][0];
      const choice = await pickPromotion(this, to, color);
      if (!choice) {
        const pieceEl = this.pieceEls.get(from);
        if (pieceEl) this.place(pieceEl, from);
        this.selected = null;
        this.renderMarks();
        return;
      }
      uci = squareName(from) + squareName(to) + choice;
    }
    this.selected = null;
    this.legal = [];
    this.renderMarks();
    this.options.onMove?.(uci);
  }
}

/** Petit menu pour choisir la pièce de promotion, posé sur la case d'arrivée. */
function pickPromotion(board, sq, color) {
  return new Promise((resolve) => {
    const promo = $("#promo");
    const rect = board.root.getBoundingClientRect();
    const size = rect.width / 8;
    const [col, row] = board.colRow(sq);
    promo.innerHTML = "";
    promo.style.left = `${rect.left + col * size}px`;
    promo.style.width = `${size}px`;
    const fromTop = row === 0;
    promo.style.top = fromTop ? `${rect.top}px` : "";
    promo.style.bottom = fromTop ? "" : `${window.innerHeight - rect.bottom}px`;
    promo.style.flexDirection = fromTop ? "column" : "column-reverse";
    const close = (value) => {
      promo.hidden = true;
      document.removeEventListener("pointerdown", outside, true);
      resolve(value);
    };
    const outside = (e) => { if (!promo.contains(e.target)) close(null); };
    for (const piece of ["q", "n", "r", "b"]) {
      const button = el("button");
      button.style.height = `${size}px`;
      button.style.backgroundImage = `url(/pieces/${color}${piece.toUpperCase()}.svg)`;
      button.setAttribute("aria-label", piece);
      button.addEventListener("click", () => close(piece));
      promo.append(button);
    }
    promo.hidden = false;
    setTimeout(() => document.addEventListener("pointerdown", outside, true));
  });
}

/* ═══════════════════════════ État de l'application ═══════════════════════════ */

const THINK_STEPS = [500, 1000, 1500, 2000, 3000, 5000, 8000, 12000];
const PIECE_ORDER = ["P", "N", "B", "R", "Q", "K"];
const PIECE_LABELS = { P: "Pion", N: "Cavalier", B: "Fou", R: "Tour", Q: "Dame", K: "Roi" };
const PIECE_POINTS = { P: 1, N: 3, B: 3, R: 5, Q: 9, K: 0 };
const START_COUNT = { P: 8, N: 2, B: 2, R: 2, Q: 1, K: 1 };

const app = {
  view: "play",
  game: null,
  orientation: "w",
  color: "white",
  thinkIndex: 3,
  train: null,
  champion: null,
  candidate: null,
  layout: null,
  heatPiece: 1,
  heatPhase: "mg",
  lastEpoch: null,
};

const board = new BoardView($("#board"), { interactive: true, onMove: playMove });
const liveBoard = new BoardView($("#live-board"));
const measureBoard = new BoardView($("#measure-board"));

/* ═══════════════════════════ Partie ═══════════════════════════ */

async function playMove(uci) {
  try {
    renderGame(await api("/api/game/move", { uci }));
  } catch (error) {
    toast(error.message);
    renderGame(await api("/api/game"));
  }
}

async function newGame() {
  const game = await api("/api/game/new", { human: app.color, think_ms: THINK_STEPS[app.thinkIndex] });
  app.orientation = game.human;
  board.setOrientation(app.orientation);
  renderGame(game);
}

function renderGame(game) {
  const previous = app.game;
  app.game = game;
  board.setPosition(game.fen, { lastMove: game.last_move, check: game.check });
  board.setLegal(game.legal);

  renderPlayers(game);
  renderMoves(game);
  renderEngine(game.engine, game.thinking);
  $("#undo").disabled = game.sans.length === 0;

  const plaque = $("#result");
  if (game.status !== "playing" && !game.thinking) {
    const humanWon = game.status === "checkmate" && game.turn !== game.human;
    const title = game.status === "checkmate" ? (humanWon ? "Vous avez gagné" : "Le moteur l’emporte") : "Partie nulle";
    $("#result-title").textContent = title;
    $("#result-detail").textContent = game.result || "";
    if (!previous || previous.id !== game.id || plaque.hidden) plaque.hidden = false;
  } else {
    plaque.hidden = true;
  }
}

/** Pièces capturées et avantage matériel, calculés depuis la position. */
function materialInfo(fen) {
  const count = { w: {}, b: {} };
  for (const code of parseFen(fen)) {
    if (!code) continue;
    count[code[0]][code[1]] = (count[code[0]][code[1]] || 0) + 1;
  }
  const captured = { w: [], b: [] };
  let points = 0;
  for (const piece of PIECE_ORDER) {
    const white = count.w[piece] || 0;
    const black = count.b[piece] || 0;
    points += (white - black) * PIECE_POINTS[piece];
    for (let i = 0; i < Math.max(0, START_COUNT[piece] - black); i++) captured.w.push("b" + piece);
    for (let i = 0; i < Math.max(0, START_COUNT[piece] - white); i++) captured.b.push("w" + piece);
  }
  return { captured, points };
}

function renderPlayers(game) {
  const material = materialInfo(game.fen);
  const bottomColor = app.orientation;
  const topColor = bottomColor === "w" ? "b" : "w";
  const generation = app.train ? app.train.generation : 0;

  const fill = (container, color) => {
    container.innerHTML = "";
    const isHuman = color === game.human;
    const chip = el("span", "player-chip " + (color === "w" ? "is-white" : "is-black"));
    const name = el("span", "player-name", isHuman ? "Vous" : "Moteur");
    container.append(chip, name);
    if (!isHuman) {
      const label = isNetwork(app.champion) ? "réseau de neurones" : generation ? "formule ajustée" : "valeurs de départ";
      container.append(el("span", "player-sub", generation ? `${label} · génération ${generation}` : label));
    }
    const captured = el("span", "player-captured");
    for (const code of material.captured[color]) {
      const img = el("img");
      img.src = `/pieces/${code}.svg`;
      img.alt = "";
      captured.append(img);
    }
    container.append(captured);
    const advantage = color === "w" ? material.points : -material.points;
    if (advantage > 0) container.append(el("span", "player-advantage", `+${advantage}`));
    if (!isHuman && game.thinking) container.append(el("span", "player-status", "réfléchit"));
  };
  fill($("#player-top"), topColor);
  fill($("#player-bottom"), bottomColor);
}

function renderMoves(game) {
  const list = $("#moves");
  list.innerHTML = "";
  if (!game.sans.length) {
    list.append(el("li", "moves-empty", "Les coups de la partie s’afficheront ici."));
    return;
  }
  for (let i = 0; i < game.sans.length; i += 2) {
    const item = el("li");
    item.append(el("span", "num", `${i / 2 + 1}.`));
    for (const j of [i, i + 1]) {
      const move = el("span", "mv", game.sans[j] || "");
      if (j === game.sans.length - 1) move.classList.add("is-current");
      item.append(move);
    }
    list.append(item);
  }
  list.scrollTop = list.scrollHeight;
}

function renderEngine(info, thinking) {
  $("#thinking-dot").hidden = !thinking;
  if (!info) {
    $("#eval-big").textContent = fmtEval(null);
    $("#eval-caption").textContent = thinking ? "le moteur réfléchit" : "en attente";
    for (const id of ["st-depth", "st-nodes", "st-nps", "st-time"]) $("#" + id).textContent = "–";
    $("#pv").textContent = "–";
    setEvalBar(null);
    return;
  }
  $("#eval-big").textContent = fmtEval(info);
  let caption;
  if (info.mate !== null && info.mate !== undefined) {
    caption = `mat en ${Math.abs(info.mate)} pour les ${info.mate > 0 ? "Blancs" : "Noirs"}`;
  } else if (Math.abs(info.cp) < 25) {
    caption = "position équilibrée";
  } else {
    caption = `avantage ${info.cp > 0 ? "Blancs" : "Noirs"}`;
  }
  $("#eval-caption").textContent = caption;
  $("#st-depth").textContent = info.depth;
  $("#st-nodes").textContent = fmtCompact(info.nodes);
  $("#st-nps").textContent = fmtCompact(info.nps) + "/s";
  $("#st-time").textContent = fmtDecimal(info.time_ms / 1000, 1) + " s";

  const pv = $("#pv");
  pv.innerHTML = "";
  let number = info.pv_start;
  let white = info.pv_turn === "w";
  info.pv.slice(0, 12).forEach((san, i) => {
    const move = el("span", "m");
    if (white) move.append(el("span", "n", `${number}.`));
    else if (i === 0) move.append(el("span", "n", `${number}…`));
    move.append(san);
    pv.append(move);
    if (!white) number += 1;
    white = !white;
  });
  setEvalBar(info);
}

function setEvalBar(info) {
  let share = 0.5;
  let label = "";
  if (info) {
    if (info.mate !== null && info.mate !== undefined) {
      share = info.mate > 0 ? 1 : 0;
      label = "M" + Math.abs(info.mate);
    } else {
      share = 1 / (1 + Math.exp(-info.cp / 250));
      label = fmtDecimal(Math.abs(info.cp) / 100, 1);
    }
  }
  const bar = $("#evalbar");
  bar.classList.toggle("is-flipped", app.orientation === "b");
  $("#evalbar-fill").style.height = `${share * 100}%`;
  const labelEl = $("#evalbar-label");
  labelEl.textContent = label;
  const whiteAhead = share >= 0.5;
  // Le chiffre s'affiche du côté du camp qui mène.
  const atBottom = (app.orientation === "w") === whiteAhead;
  labelEl.classList.toggle("is-top", !atBottom);
}

/* ═══════════════════════════ Entraînement ═══════════════════════════ */

const PHASES = ["selfplay", "tuning", "match", "verify"];

async function loadTraining() {
  const data = await api("/api/train");
  app.train = data.state;
  app.champion = data.champion;
  app.candidate = data.candidate || app.candidate;
  app.layout = data.layout;
  $("#cfg-games").value = String(data.state.config.games_per_generation);
  $("#cfg-nodes").value = String(data.state.config.nodes_per_move);
  renderTraining();
}

function renderTraining() {
  const state = app.train;
  if (!state) return;
  const running = state.running;

  $("#kpi-gen").textContent = fmtInt(state.generation);
  const elo = $("#kpi-elo");
  elo.textContent = fmtSigned(state.elo);
  elo.classList.toggle("is-positive", state.elo >= 1);
  $("#kpi-games").textContent = fmtInt(state.total_games);
  $("#kpi-data").textContent = fmtInt(state.dataset);

  const toggle = $("#train-toggle");
  toggle.textContent = running ? "Arrêter" : state.generation ? "Reprendre" : "Démarrer";
  $("#cfg-games").disabled = running;
  $("#cfg-nodes").disabled = running;
  $("#reset-toggle").disabled = running;

  const activeIndex = running ? PHASES.indexOf(state.phase) : -1;
  $$("#pipeline li").forEach((item, i) => {
    item.classList.toggle("is-active", i === activeIndex);
    item.classList.toggle("is-done", activeIndex > i);
    const bar = $("i", item);
    bar.style.width = i === activeIndex ? `${Math.round(state.progress * 100)}%` : "";
  });

  const origin = state.origin === "zero" ? "départ : zéro absolu" : "départ : valeurs classiques";
  $("#origin-note").textContent = state.mode === "nnue" ? `${origin} · puis réseau de neurones` : origin;
  $$("#pipeline .step-name")[1].textContent = state.mode === "nnue" ? "Réseau" : "Ajustement";
  renderPhaseLine();
  renderStatus();
  drawEloChart();
  drawScoreChart();
  drawLossChart();
  renderWdl();
  renderWeights();
  renderLog();
}

function renderStatus() {
  const state = app.train;
  const status = $("#topbar-status");
  if (!state) return;
  const kind = isNetwork(app.champion) ? "réseau" : "formule";
  const parts = [`Champion <b>génération ${state.generation}</b> (${kind})`];
  if (state.generation) parts.push(`<b>${fmtSigned(state.elo)}</b> ELO`);
  if (state.running) parts.push("entraînement en cours");
  status.innerHTML = parts.join(" · ");
}

function wdlCount(wdl) {
  return wdl.wins + wdl.draws + wdl.losses;
}

function renderPhaseLine() {
  const state = app.train;
  const line = $("#phase-line");
  const next = state.generation + 1;
  const games = state.config.games_per_generation;
  const matchGames = state.config.match_games;
  if (!state.running) {
    line.innerHTML = state.generation
      ? `En pause après la génération ${state.generation}. Le champion joue les parties de l’onglet Partie.`
      : "Prêt. Lance l’entraînement et regarde le moteur apprendre, génération après génération.";
    return;
  }
  const s = state.selfplay;
  const m = state.matchup;
  const v = state.verification;
  switch (state.phase) {
    case "selfplay":
      if (state.mode === "nnue" && state.dataset < state.collect_target) {
        line.innerHTML = `Collecte des données pour le réseau : <b>${fmtInt(state.dataset)}</b> / ${fmtInt(state.collect_target)} positions · parties en cours <b>${wdlCount(s)}</b> / ${games}`;
      } else {
        line.innerHTML = `Génération ${next} · le champion joue contre lui-même : <b>${wdlCount(s)}</b> / ${games} parties`;
      }
      break;
    case "tuning": {
      const epoch = app.lastEpoch ? `, passe <b>${app.lastEpoch.epoch}</b> / ${app.lastEpoch.epochs}` : "";
      line.innerHTML = state.mode === "nnue"
        ? `Génération ${next} · entraînement du réseau sur <b>${fmtInt(state.dataset)}</b> positions${epoch}`
        : `Génération ${next} · ajustement des poids sur <b>${fmtInt(state.dataset)}</b> positions${epoch.replace("passe", "itération")}`;
      break;
    }
    case "match":
      line.innerHTML = `Génération ${next} · match de sélection, candidat contre champion : <b>${wdlCount(m)}</b> / ${matchGames} parties`;
      break;
    case "verify":
      line.innerHTML = `Génération ${next} · le candidat a gagné la sélection, il doit confirmer sur de nouvelles ouvertures : <b>${wdlCount(v)}</b> / ${matchGames}`;
      break;
    default:
      line.textContent = "";
  }
}

function renderWdl() {
  const state = app.train;
  const box = $("#live-wdl");
  const phase = state.running ? state.phase : "selfplay";
  let wdl = state.selfplay;
  let labels = ["Gains blancs", "Nulles", "Gains noirs"];
  if (phase === "match" || phase === "verify") {
    wdl = phase === "match" ? state.matchup : state.verification;
    labels = ["Candidat", "Nulles", "Champion"];
  }
  const total = Math.max(1, wdlCount(wdl));
  box.innerHTML = "";
  [wdl.wins, wdl.draws, wdl.losses].forEach((value, i) => {
    const cell = el("div");
    cell.append(el("dt", "", labels[i]), el("dd", "", fmtInt(value)));
    box.append(cell);
  });
  const bar = el("div", "wdl-bar");
  for (const [cls, value] of [["w", wdl.wins], ["d", wdl.draws], ["l", wdl.losses]]) {
    const part = el("i", cls);
    part.style.width = `${(value / total) * 100}%`;
    bar.append(part);
  }
  box.append(bar);
}

function onLive(message) {
  liveBoard.setPosition(message.fen, {
    lastMove: message.last_move ? [message.last_move.slice(0, 2), message.last_move.slice(2, 4)] : null,
  });
  $("#live-white").textContent = `Blancs · ${message.white}`;
  $("#live-black").textContent = `Noirs · ${message.black}`;
  const label = message.phase === "selfplay" ? "entraînement" : message.phase === "match" ? "match" : "vérification";
  $("#live-note").textContent = `${label} · partie ${message.game} · coup ${Math.max(1, Math.ceil(message.ply / 2))}`;
}

/* ─────────── Poids : valeurs, carte de chaleur, critères ─────────── */

/**
 * Évaluation à montrer : le candidat pendant qu'on l'entraîne et qu'il joue
 * ses matchs ; en mode réseau, le dernier réseau entraîné tant que le
 * champion est encore la formule classique ; sinon le champion.
 */
function displayedWeights() {
  const state = app.train;
  const testing = state && state.running && ["tuning", "match", "verify"].includes(state.phase);
  const networkPending = state && state.mode === "nnue" && !isNetwork(app.champion);
  if (app.candidate && (testing || (networkPending && isNetwork(app.candidate)))) {
    return { weights: app.candidate, candidate: true };
  }
  return { weights: app.champion, candidate: false };
}

const isNetwork = (view) => view && view.kind === "nnue";

/** Valeur (milieu, finale) des cinq pièces, quel que soit le type d'évaluation. */
function pieceValues(view) {
  if (isNetwork(view)) {
    return [0, 1, 2, 3, 4].map((i) => [view.probe.mg.values[i], view.probe.eg.values[i]]);
  }
  const base = app.layout.material;
  return [0, 1, 2, 3, 4].map((i) => [view.mg[base + i], view.eg[base + i]]);
}

/** Bonus de chaque case (a1 = 0) pour un type de pièce ; null si la case n'a pas de sens. */
function squareTable(view, phase, kind) {
  if (isNetwork(view)) return view.probe[phase].pst[kind];
  const offset = app.layout.pst + kind * 64;
  return view[phase].slice(offset, offset + 64);
}

function renderWeights() {
  if (!app.champion || !app.layout) return;
  const { weights, candidate } = displayedWeights();
  const network = isNetwork(weights);
  const phase = app.train && app.train.running ? app.train.phase : "idle";
  let note = network ? "réseau champion" : "champion";
  if (candidate) {
    if (phase === "tuning") note = network ? "réseau en cours d’entraînement" : "candidat en cours d’ajustement";
    else if (phase === "match" || phase === "verify") note = network ? "réseau candidat, en match" : "candidat, en match";
    else note = "dernier réseau entraîné";
  }
  $("#values-note").textContent = note;
  $(".tile-values .caption").textContent = network
    ? "Mesurée en interrogeant le réseau : ce que rapporte une pièce blanche ajoutée à une position de référence, en moyenne sur toutes les cases. En centipions (100 = un pion)."
    : "En centipions (100 = un pion), en milieu de partie et en finale.";
  $(".tile-heat .caption").textContent = network
    ? "Ce que le réseau pense de chaque case : bonus (or) ou malus (rouille) par rapport à la moyenne. Milieu : rois et pions en place. Finale : rois seuls."
    : "Bonus (or) ou malus (rouille) selon la case, vus du côté des Blancs.";
  $(".tile-terms .eyebrow").textContent = network ? "Le réseau" : "Autres critères";
  renderValues(weights, candidate);
  renderHeatmap(weights);
  if (network) renderNetwork(weights);
  else renderTerms(weights);
}

function renderValues(weights, candidate) {
  const box = $("#values");
  const values = pieceValues(weights);
  const reference = candidate && app.champion.kind === weights.kind ? pieceValues(app.champion) : null;
  const scale = Math.max(1000, ...values.flat().map(Math.abs));
  box.innerHTML = "";
  const head = el("div", "value-head");
  head.append(el("span"), el("span"), el("span", "", "Milieu"), el("span", "", "Finale"));
  box.append(head);
  values.forEach(([mg, eg], i) => {
    const row = el("div", "value-row");
    const img = el("img");
    img.src = `/pieces/w${PIECE_ORDER[i]}.svg`;
    img.alt = PIECE_LABELS[PIECE_ORDER[i]];
    img.title = PIECE_LABELS[PIECE_ORDER[i]];
    const bars = el("div", "value-bars");
    for (const [cls, value] of [["mg", mg], ["eg", eg]]) {
      const bar = el("div", "value-bar " + cls);
      const fill = el("i");
      fill.style.width = `${(Math.max(0, value) / scale) * 100}%`;
      bar.append(fill);
      bars.append(bar);
    }
    const numbers = [mg, eg].map((value, j) => {
      const num = el("span", "value-num" + (j ? " eg" : ""), fmtInt(value));
      if (reference) {
        if (value > reference[i][j]) num.classList.add("delta-up");
        if (value < reference[i][j]) num.classList.add("delta-down");
      }
      return num;
    });
    row.append(img, bars, ...numbers);
    box.append(row);
  });
}

function renderHeatPieces() {
  const box = $("#heat-pieces");
  box.innerHTML = "";
  PIECE_ORDER.forEach((piece, i) => {
    const button = el("button");
    button.title = PIECE_LABELS[piece];
    button.classList.toggle("is-active", i === app.heatPiece);
    const img = el("img");
    img.src = `/pieces/w${piece}.svg`;
    img.alt = PIECE_LABELS[piece];
    button.append(img);
    button.addEventListener("click", () => {
      app.heatPiece = i;
      renderHeatPieces();
      renderWeights();
    });
    box.append(button);
  });
}

function renderHeatmap(weights) {
  const map = $("#heatmap");
  const values = squareTable(weights, app.heatPhase, app.heatPiece);
  const peak = Math.max(20, ...values.filter((v) => v !== null).map(Math.abs));
  map.innerHTML = "";
  for (let rank = 7; rank >= 0; rank--) {
    map.append(el("span", "heat-axis", String(rank + 1)));
    for (let file = 0; file < 8; file++) {
      const value = values[rank * 8 + file];
      if (value === null || value === undefined) {
        const cell = el("span", "heat-cell is-empty");
        cell.title = `${FILES[file]}${rank + 1} : case occupée dans la position de référence`;
        map.append(cell);
        continue;
      }
      const cell = el("span", "heat-cell", value ? String(value) : "");
      const alpha = 0.06 + 0.8 * Math.min(1, Math.abs(value) / peak);
      cell.style.backgroundColor = value >= 0
        ? `rgba(205, 184, 131, ${value === 0 ? 0.04 : alpha})`
        : `rgba(179, 106, 82, ${alpha})`;
      cell.style.color = Math.abs(value) / peak > 0.55 ? "#101312" : "";
      cell.title = `${FILES[file]}${rank + 1} : ${value}`;
      map.append(cell);
    }
  }
  map.append(el("span"));
  for (const file of FILES) map.append(el("span", "heat-axis", file));
}

function renderNetwork(view) {
  const box = $("#terms");
  box.innerHTML = "";
  const validation = app.train && app.train.history.length
    ? [...app.train.history].reverse().find((g) => g.mode === "nnue")
    : null;
  const rows = [
    ["Entrées", `${view.inputs}`],
    ["Neurones cachés", `2 × ${view.hidden}`],
    ["Paramètres", fmtInt(view.parameters)],
    ["Passes d’entraînement", fmtInt(view.epochs)],
    ["Erreur de validation", validation ? fmtDecimal(validation.loss_after, 4) : "–"],
  ];
  for (const [name, value] of rows) {
    const row = el("div", "term-row term-row-2");
    row.append(el("span", "term-name", name), el("span", "term-val", value));
    box.append(row);
  }
  box.append(el("p", "caption",
    "Chaque entrée signale une pièce sur une case (2 couleurs × 6 pièces × 64 cases). "
    + "Les neurones cachés combinent ces entrées ; la sortie donne l’évaluation. "
    + "Personne n’a écrit ces règles : le réseau les a déduites des parties."));
}

function renderTerms(weights) {
  const L = app.layout;
  const box = $("#terms");
  box.innerHTML = "";

  box.append(el("span", "pv-label", "Pion passé, selon sa rangée"));
  const passed = el("div", "passed");
  const labels = el("div", "passed-labels");
  const passedValues = [1, 2, 3, 4, 5, 6].map((r) => (weights.mg[L.passed_pawn + r] + weights.eg[L.passed_pawn + r]) / 2);
  const peak = Math.max(10, ...passedValues);
  passedValues.forEach((value, i) => {
    const bar = el("i");
    bar.style.height = `${(Math.max(0, value) / peak) * 100}%`;
    bar.title = `${i + 2}e rangée : ${Math.round(value)}`;
    passed.append(bar);
    labels.append(el("span", "", `${i + 2}e`));
  });
  box.append(passed, labels);

  const head = el("div", "term-row term-head");
  head.append(el("span", "", "Critère"), el("span", "", "Milieu"), el("span", "", "Finale"));
  box.append(head);
  const rows = [
    ["Pion doublé", L.doubled_pawn],
    ["Pion isolé", L.isolated_pawn],
    ["Tour, colonne ouverte", L.rook_open_file],
    ["Tour, semi-ouverte", L.rook_semi_open_file],
    ["Paire de fous", L.bishop_pair],
    ["Mobilité du cavalier", L.mobility],
    ["Mobilité du fou", L.mobility + 1],
    ["Mobilité de la tour", L.mobility + 2],
    ["Mobilité de la dame", L.mobility + 3],
    ["Bouclier de pions du roi", L.king_shield],
    ["Avoir le trait", L.tempo],
  ];
  for (const [name, index] of rows) {
    const row = el("div", "term-row");
    row.append(
      el("span", "term-name", name),
      el("span", "term-val", fmtSigned(weights.mg[index])),
      el("span", "term-val eg", fmtSigned(weights.eg[index])),
    );
    box.append(row);
  }
}

function renderLog() {
  const body = $("#log");
  const history = app.train.history;
  body.innerHTML = "";
  if (!history.length) {
    const row = el("tr", "log-empty");
    const cell = el("td", "", "Aucune génération pour l’instant.");
    cell.colSpan = 9;
    row.append(cell);
    body.append(row);
    return;
  }
  const score = (w) => `${w.wins} – ${w.draws} – ${w.losses}`;
  for (const g of [...history].reverse()) {
    const row = el("tr");
    const cells = [
      [g.mode === "nnue" ? `${g.index} · réseau` : String(g.index)],
      [fmtInt(g.games)],
      [`+${fmtInt(g.samples)}`, "muted"],
      [`${fmtDecimal(g.loss_before, 4)} → ${fmtDecimal(g.loss_after, 4)}`, "muted"],
      [score(g.matchup)],
      [g.verification ? score(g.verification) : "—", g.verification ? "" : "muted"],
      [g.accepted ? fmtSigned(g.elo_gain) : "—", g.accepted ? "" : "muted"],
      [`${fmtDecimal(g.seconds, 1)} s`, "muted"],
    ];
    for (const [text, cls] of cells) row.append(el("td", cls || "", text));
    const tagCell = el("td");
    const tag = g.accepted ? "Adopté" : g.verification ? "Non confirmé" : "Rejeté";
    tagCell.append(el("span", "tag " + (g.accepted ? "tag-ok" : "tag-no"), tag));
    row.append(tagCell);
    body.append(row);
  }
}

/* ─────────── Graphiques (SVG dessiné à la main) ─────────── */

function niceStep(range, count) {
  const raw = range / Math.max(1, count);
  const power = Math.pow(10, Math.floor(Math.log10(raw)));
  const fraction = raw / power;
  const nice = fraction <= 1 ? 1 : fraction <= 2 ? 2 : fraction <= 5 ? 5 : 10;
  return nice * power;
}

function ticks(min, max, count) {
  const step = niceStep(max - min || 1, count);
  const result = [];
  for (let v = Math.ceil(min / step) * step; v <= max + step * 1e-9; v += step) result.push(+v.toFixed(10));
  return result;
}

function chartBox(container, padding = { left: 44, right: 12, top: 10, bottom: 22 }) {
  const width = container.clientWidth || 600;
  const height = container.clientHeight || 200;
  return {
    width,
    height,
    left: padding.left,
    right: width - padding.right,
    top: padding.top,
    bottom: height - padding.bottom,
  };
}

function attachTooltip(container, points, text) {
  let tip = $(".chart-tip", container);
  const svg = $("svg", container);
  if (!points.length || !svg) return;
  svg.addEventListener("pointermove", (e) => {
    const rect = container.getBoundingClientRect();
    const x = e.clientX - rect.left;
    let best = points[0];
    for (const p of points) if (Math.abs(p.px - x) < Math.abs(best.px - x)) best = p;
    if (!tip) {
      tip = el("div", "chart-tip");
      container.append(tip);
    }
    tip.hidden = false;
    tip.textContent = text(best);
    tip.style.left = `${best.px}px`;
    tip.style.top = `${best.py}px`;
  });
  svg.addEventListener("pointerleave", () => { if (tip) tip.hidden = true; });
}

function drawEloChart() {
  const container = $("#chart-elo");
  const history = app.train.history;
  const box = chartBox(container);
  const points = [{ x: 0, y: 0, accepted: true, gen: 0 }].concat(
    history.map((g) => ({
      x: g.index,
      y: g.elo,
      accepted: g.accepted,
      confirmed: g.verification ? g.accepted : undefined,
      gen: g.index,
      gain: g.elo_gain,
    })),
  );
  const maxX = Math.max(5, points[points.length - 1].x);
  let minY = Math.min(0, ...points.map((p) => p.y));
  let maxY = Math.max(20, ...points.map((p) => p.y));
  const yTicks = ticks(minY, maxY, 4);
  minY = Math.min(minY, yTicks[0]);
  maxY = Math.max(maxY, yTicks[yTicks.length - 1]);
  const sx = (x) => box.left + (x / maxX) * (box.right - box.left);
  const sy = (y) => box.bottom - ((y - minY) / (maxY - minY || 1)) * (box.bottom - box.top);

  let svg = `<svg viewBox="0 0 ${box.width} ${box.height}"><defs><linearGradient id="fade" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#cdb883" stop-opacity=".14"/><stop offset="1" stop-color="#cdb883" stop-opacity="0"/></linearGradient></defs>`;
  for (const t of yTicks) {
    svg += `<line class="${t === 0 ? "zero-line" : "grid-line"}" x1="${box.left}" x2="${box.right}" y1="${sy(t)}" y2="${sy(t)}"/>`;
    svg += `<text class="axis-label" x="${box.left - 10}" y="${sy(t) + 3}" text-anchor="end">${fmtSigned(t)}</text>`;
  }
  for (const t of ticks(0, maxX, 6).filter((t) => Number.isInteger(t))) {
    svg += `<text class="axis-label" x="${sx(t)}" y="${box.bottom + 16}" text-anchor="middle">${t}</text>`;
  }
  if (history.length) {
    const line = points.map((p, i) => `${i ? "L" : "M"}${sx(p.x).toFixed(1)},${sy(p.y).toFixed(1)}`).join("");
    svg += `<path class="series-area" d="${line}L${sx(points[points.length - 1].x)},${box.bottom}L${sx(0)},${box.bottom}Z"/>`;
    svg += `<path class="series" d="${line}"/>`;
    for (const p of points.slice(1)) {
      svg += p.accepted
        ? `<circle class="dot" cx="${sx(p.x)}" cy="${sy(p.y)}" r="2.6"/>`
        : `<circle class="dot-rejected" cx="${sx(p.x)}" cy="${sy(p.y)}" r="2.2"/>`;
    }
  } else {
    svg += `<text class="empty" x="${(box.left + box.right) / 2}" y="${(box.top + box.bottom) / 2}" text-anchor="middle">La courbe apparaîtra après la première génération.</text>`;
  }
  svg += "</svg>";
  container.innerHTML = svg;
  const hover = points.slice(1).map((p) => ({ ...p, px: sx(p.x), py: sy(p.y) }));
  attachTooltip(container, hover, (p) =>
    `Génération ${p.gen} · ${fmtSigned(p.y)} ELO · ${p.accepted ? `adoptée (${fmtSigned(p.gain)})` : p.confirmed === false ? "non confirmée" : "rejetée"}`);
}

function drawScoreChart() {
  const container = $("#chart-score");
  const history = app.train.history;
  const box = chartBox(container, { left: 44, right: 12, top: 4, bottom: 4 });
  const maxX = Math.max(5, history.length ? history[history.length - 1].index : 0);
  const mid = (box.top + box.bottom) / 2;
  const half = (box.bottom - box.top) / 2;
  const sx = (x) => box.left + (x / maxX) * (box.right - box.left);
  const barWidth = Math.max(2, Math.min(8, ((box.right - box.left) / maxX) * 0.55));
  let svg = `<svg viewBox="0 0 ${box.width} ${box.height}">`;
  svg += `<line class="zero-line" x1="${box.left}" x2="${box.right}" y1="${mid}" y2="${mid}"/>`;
  svg += `<text class="axis-label" x="${box.left - 10}" y="${mid + 3}" text-anchor="end">50 %</text>`;
  const hover = [];
  for (const g of history) {
    const total = Math.max(1, wdlCount(g.matchup));
    const score = (g.matchup.wins + g.matchup.draws / 2) / total;
    const height = Math.max(1, Math.abs(score - 0.5) * 2 * half);
    const y = score >= 0.5 ? mid - height : mid;
    svg += `<rect class="${g.accepted ? "bar-accepted" : "bar-rejected"}" x="${sx(g.index) - barWidth / 2}" y="${y}" width="${barWidth}" height="${height}"/>`;
    hover.push({ px: sx(g.index), py: y, g, score });
  }
  svg += "</svg>";
  container.innerHTML = svg;
  attachTooltip(container, hover, (p) =>
    `Génération ${p.g.index} · candidat ${Math.round(p.score * 100)} % (${p.g.matchup.wins} – ${p.g.matchup.draws} – ${p.g.matchup.losses})`);
}

function drawLossChart() {
  const container = $("#chart-loss");
  const state = app.train;
  const box = chartBox(container, { left: 52, right: 12, top: 10, bottom: 22 });
  let values = state.losses;
  let validation = state.validation_losses || [];
  let xLabel = state.mode === "nnue" ? "lot" : "itération";
  let note = "";
  if (values.length) {
    note = validation.length
      ? `validation ${fmtDecimal(validation[validation.length - 1][1], 4)}`
      : `${fmtDecimal(values[0], 4)} → ${fmtDecimal(values[values.length - 1], 4)}`;
  } else if (state.history.length) {
    values = state.history.map((g) => g.loss_after);
    validation = [];
    xLabel = "génération";
    note = "après chaque génération";
  }
  $("#loss-note").textContent = note || "–";

  let svg = `<svg viewBox="0 0 ${box.width} ${box.height}">`;
  if (values.length >= 2) {
    // La toute première chute est très forte : on cadre sur la suite pour
    // qu'on voie la progression (le début sort par le haut).
    const settled = values.length > 20 ? values.slice(Math.floor(values.length * 0.08)) : values;
    const all = settled.concat(validation.map((p) => p[1]));
    let min = Math.min(...all);
    let max = Math.max(...all);
    const pad = (max - min) * 0.1 || max * 0.02 || 0.01;
    min -= pad;
    max += pad;
    const sx = (i) => box.left + (i / (values.length - 1)) * (box.right - box.left);
    const sy = (v) => box.bottom - ((Math.min(v, max) - min) / (max - min)) * (box.bottom - box.top);
    for (const t of ticks(min, max, 3)) {
      if (t < min || t > max) continue;
      svg += `<line class="grid-line" x1="${box.left}" x2="${box.right}" y1="${sy(t)}" y2="${sy(t)}"/>`;
      svg += `<text class="axis-label" x="${box.left - 10}" y="${sy(t) + 3}" text-anchor="end">${fmtDecimal(t, 4)}</text>`;
    }
    svg += `<text class="axis-label" x="${box.right}" y="${box.bottom + 16}" text-anchor="end">${xLabel} ${values.length}</text>`;
    svg += `<text class="axis-label" x="${box.left}" y="${box.bottom + 16}">1</text>`;
    const line = values.map((v, i) => `${i ? "L" : "M"}${sx(i).toFixed(1)},${sy(v).toFixed(1)}`).join("");
    svg += `<path class="series" d="${line}"/>`;
    if (validation.length) {
      const vline = validation.map(([i, v], k) => `${k ? "L" : "M"}${sx(i).toFixed(1)},${sy(v).toFixed(1)}`).join("");
      svg += `<path class="series-validation" d="${vline}"/>`;
      for (const [i, v] of validation) svg += `<circle class="dot-validation" cx="${sx(i)}" cy="${sy(v)}" r="2.6"/>`;
    }
  } else {
    svg += `<text class="empty" x="${(box.left + box.right) / 2}" y="${(box.top + box.bottom) / 2}" text-anchor="middle">En attente de la première phase d’apprentissage.</text>`;
  }
  svg += "</svg>";
  container.innerHTML = svg;
}

/* ═══════════════════════════ Mesure ═══════════════════════════ */

async function loadMeasure() {
  app.measure = await api("/api/measure");
  renderMeasure();
}

/** La mesure affichée : celle en cours, sinon la dernière terminée. */
function shownRun() {
  const state = app.measure;
  if (!state) return null;
  return state.current || state.history[state.history.length - 1] || null;
}

const played = (wdl) => wdl.wins + wdl.draws + wdl.losses;
const fmtClock = (ms) => {
  const seconds = Math.max(0, Math.ceil(ms / 1000));
  return `${Math.floor(seconds / 60)}:${String(seconds % 60).padStart(2, "0")}`;
};
const fmtCadence = (run) => `${run.base_ms / 1000} s + ${fmtDecimal(run.increment_ms / 1000, 1)}`;

function renderMeasure() {
  const state = app.measure;
  if (!state) return;
  const run = shownRun();
  const running = state.running;

  const status = $("#sf-status");
  status.classList.toggle("is-error", !state.stockfish);
  if (state.stockfish) {
    status.innerHTML = `Adversaire : <b>${state.stockfish.name}</b> · ${state.stockfish.path}`;
  } else {
    status.textContent = state.error || "Recherche de Stockfish…";
  }
  $("#sf-missing").hidden = Boolean(state.stockfish) || running;
  if (state.path && !$("#sf-path").value) $("#sf-path").value = state.path;

  const toggle = $("#m-toggle");
  toggle.textContent = running ? "Arrêter" : "Démarrer";
  toggle.disabled = !running && !state.stockfish;
  for (const id of ["#m-elo", "#m-count", "#m-tc"]) $(id).disabled = running;
  // Pendant une mesure, les réglages affichés sont ceux de la mesure en cours.
  if (running && state.current) {
    const current = state.current;
    const setIfPresent = (id, value) => {
      if ([...$(id).options].some((o) => o.value === value)) $(id).value = value;
    };
    setIfPresent("#m-elo", String(current.opponent_elo));
    setIfPresent("#m-count", String(current.games));
    setIfPresent("#m-tc", `${current.base_ms}/${current.increment_ms}`);
  }

  if (run) {
    const n = played(run.wdl);
    $("#m-estimate").textContent = run.estimate ? `≈ ${fmtInt(run.estimate)}` : "–";
    $("#m-margin").textContent = run.margin ? `± ${fmtInt(run.margin)}` : "–";
    $("#m-games").textContent = `${n} / ${run.games}`;
    $("#m-score").textContent = n ? `${Math.round(((run.wdl.wins + run.wdl.draws / 2) / n) * 100)} %` : "–";
  } else {
    for (const id of ["#m-estimate", "#m-margin", "#m-score"]) $(id).textContent = "–";
    $("#m-games").textContent = "0";
  }

  const line = $("#m-line");
  if (!run) {
    line.textContent = state.stockfish
      ? "Choisis un niveau et lance la mesure. Commence vers 2000, puis ajuste : l’estimation est la plus précise quand le score est proche de 50 %."
      : "";
  } else {
    const n = played(run.wdl);
    const who = `Ton moteur (${run.engine}) contre ${run.opponent} réglé à ${run.opponent_elo}, ${fmtCadence(run)}`;
    let advice = "";
    if (n >= 10 && run.wdl.wins === n) advice = " Ton moteur gagne tout : choisis un niveau plus élevé pour une mesure précise.";
    else if (n >= 10 && run.wdl.losses === n) advice = " Ton moteur perd tout : choisis un niveau plus bas pour une mesure précise.";
    else if (n >= 20) {
      const score = (run.wdl.wins + run.wdl.draws / 2) / n;
      if (score > 0.8) advice = " Score très élevé : un niveau plus haut donnerait une mesure plus précise.";
      if (score < 0.2) advice = " Score très bas : un niveau plus bas donnerait une mesure plus précise.";
    }
    const [ours, theirs] = run.time_losses || [0, 0];
    if (ours || theirs) advice += ` Pertes au temps : ton moteur ${ours}, Stockfish ${theirs}.`;
    line.innerHTML = running
      ? `${who} · <b>${n}</b> / ${run.games} parties terminées.${advice}`
      : `${run.finished ? "Mesure terminée" : "Mesure interrompue"} · ${who} · ${n} parties.${advice}`;
  }
  $("#m-note").textContent = run ? `Stockfish ${run.opponent_elo}` : "";

  renderMeasureWdl(run);
  drawMeasureChart();
  renderMeasureLog();
}

function renderMeasureWdl(run) {
  const box = $("#m-wdl");
  const wdl = run ? run.wdl : { wins: 0, draws: 0, losses: 0 };
  const total = Math.max(1, played(wdl));
  box.innerHTML = "";
  [["Victoires", wdl.wins], ["Nulles", wdl.draws], ["Défaites", wdl.losses]].forEach(([label, value]) => {
    const cell = el("div");
    cell.append(el("dt", "", label), el("dd", "", fmtInt(value)));
    box.append(cell);
  });
  const bar = el("div", "wdl-bar");
  for (const [cls, value] of [["w", wdl.wins], ["d", wdl.draws], ["l", wdl.losses]]) {
    const part = el("i", cls);
    part.style.width = `${(value / total) * 100}%`;
    bar.append(part);
  }
  box.append(bar);
}

function drawMeasureChart() {
  const container = $("#chart-measure");
  const run = shownRun();
  const box = chartBox(container, { left: 52, right: 12, top: 12, bottom: 22 });
  let svg = `<svg viewBox="0 0 ${box.width} ${box.height}">`;
  const points = run ? run.points : [];
  if (points.length) {
    const reference = run.opponent_elo;
    const lows = points.map((p) => p[1] - p[2]);
    const highs = points.map((p) => p[1] + p[2]);
    let min = Math.min(reference - 100, ...lows);
    let max = Math.max(reference + 100, ...highs);
    const yTicks = ticks(min, max, 4);
    min = Math.min(min, yTicks[0]);
    max = Math.max(max, yTicks[yTicks.length - 1]);
    const maxX = Math.max(run.games, 2);
    const sx = (x) => box.left + (x / maxX) * (box.right - box.left);
    const sy = (y) => box.bottom - ((y - min) / (max - min)) * (box.bottom - box.top);
    for (const t of yTicks) {
      svg += `<line class="grid-line" x1="${box.left}" x2="${box.right}" y1="${sy(t)}" y2="${sy(t)}"/>`;
      svg += `<text class="axis-label" x="${box.left - 10}" y="${sy(t) + 3}" text-anchor="end">${fmtInt(t)}</text>`;
    }
    for (const t of ticks(0, maxX, 5).filter(Number.isInteger)) {
      svg += `<text class="axis-label" x="${sx(t)}" y="${box.bottom + 16}" text-anchor="middle">${t}</text>`;
    }
    svg += `<line class="reference" x1="${box.left}" x2="${box.right}" y1="${sy(reference)}" y2="${sy(reference)}"/>`;
    svg += `<text class="reference-label" x="${box.right}" y="${sy(reference) - 6}" text-anchor="end">Stockfish ${reference}</text>`;
    const upper = points.map((p, i) => `${i ? "L" : "M"}${sx(p[0]).toFixed(1)},${sy(p[1] + p[2]).toFixed(1)}`).join("");
    const lower = [...points].reverse().map((p) => `L${sx(p[0]).toFixed(1)},${sy(p[1] - p[2]).toFixed(1)}`).join("");
    svg += `<path class="band" d="${upper}${lower}Z"/>`;
    const line = points.map((p, i) => `${i ? "L" : "M"}${sx(p[0]).toFixed(1)},${sy(p[1]).toFixed(1)}`).join("");
    svg += `<path class="series" d="${line}"/>`;
    const last = points[points.length - 1];
    svg += `<circle class="dot" cx="${sx(last[0])}" cy="${sy(last[1])}" r="3"/>`;
  } else {
    svg += `<text class="empty" x="${(box.left + box.right) / 2}" y="${(box.top + box.bottom) / 2}" text-anchor="middle">L’estimation apparaîtra après la première partie.</text>`;
  }
  svg += "</svg>";
  container.innerHTML = svg;
  if (points.length) {
    const maxX = Math.max(run.games, 2);
    const hover = points.map((p) => ({
      px: box.left + (p[0] / maxX) * (box.right - box.left),
      py: 20,
      p,
    }));
    attachTooltip(container, hover, ({ p }) => `Après ${p[0]} parties : ≈ ${fmtInt(p[1])} ± ${fmtInt(p[2])}`);
  }
}

function renderMeasureLog() {
  const body = $("#m-log");
  const history = app.measure.history;
  body.innerHTML = "";
  if (!history.length) {
    const row = el("tr", "log-empty");
    const cell = el("td", "", "Aucune mesure pour l’instant.");
    cell.colSpan = 7;
    row.append(cell);
    body.append(row);
    return;
  }
  for (const run of [...history].reverse()) {
    const date = new Date(run.started_at * 1000).toLocaleString("fr-FR", {
      day: "numeric",
      month: "short",
      hour: "2-digit",
      minute: "2-digit",
    });
    const n = played(run.wdl);
    const row = el("tr");
    const cells = [
      [date, "muted"],
      [run.engine],
      [`${run.opponent} · ${run.opponent_elo}`],
      [fmtCadence(run), "muted"],
      [run.finished ? String(n) : `${n} / ${run.games}`],
      [`${run.wdl.wins} – ${run.wdl.draws} – ${run.wdl.losses}`],
      [run.estimate ? `≈ ${fmtInt(run.estimate)} ± ${fmtInt(run.margin)}` : "–"],
    ];
    for (const [text, cls] of cells) row.append(el("td", cls || "", text));
    body.append(row);
  }
}

function onMeasureLive(message) {
  const last = message.last_move;
  measureBoard.setPosition(message.fen, { lastMove: last ? [last.slice(0, 2), last.slice(2, 4)] : null });
  $("#m-white").textContent = `Blancs · ${message.white}`;
  $("#m-black").textContent = `Noirs · ${message.black}`;
  $("#m-white-clock").textContent = fmtClock(message.clocks[0]);
  $("#m-black-clock").textContent = fmtClock(message.clocks[1]);
  $("#m-live-note").textContent = `partie ${message.game} · coup ${Math.max(1, Math.ceil(message.ply / 2))}`;
}

/* ═══════════════════════════ Événements en direct ═══════════════════════════ */

const handlers = {
  game(message) {
    renderGame(message.game);
  },
  think(message) {
    if (!app.game || message.id !== app.game.id) return;
    app.game.engine = message.info;
    renderEngine(message.info, app.game.thinking);
  },
  train_state(message) {
    const wasRunning = app.train && app.train.running;
    app.train = message.state;
    if (message.state.phase !== "tuning") app.lastEpoch = null;

    renderTraining();
    if (app.game) renderPlayers(app.game);
  },
  train_progress(message) {
    if (!app.train) return;
    Object.assign(app.train, {
      phase: message.phase,
      progress: message.progress,
      selfplay: message.selfplay,
      matchup: message.matchup,
      verification: message.verification,
      dataset: message.dataset,
    });
    $("#kpi-data").textContent = fmtInt(message.dataset);
    renderPhaseLine();
    renderWdl();
    const index = PHASES.indexOf(message.phase);
    const item = $$("#pipeline li")[index];
    if (item) $("i", item).style.width = `${Math.round(message.progress * 100)}%`;
  },
  train_epoch(message) {
    if (!app.train) return;
    app.lastEpoch = message;
    app.train.losses.push(message.loss);
    if (message.validation !== null && message.validation !== undefined) {
      app.train.validation_losses.push([app.train.losses.length - 1, message.validation]);
    }
    app.train.progress = message.steps ? message.step / message.steps : message.epoch / message.epochs;
    const item = $$("#pipeline li")[1];
    if (item) $("i", item).style.width = `${Math.round(app.train.progress * 100)}%`;
    if (message.candidate) {
      app.candidate = message.candidate;
      renderWeights();
    }
    renderPhaseLine();
    drawLossChart();
  },
  train_weights(message) {
    app.champion = message.champion;
    app.candidate = message.candidate;
    renderWeights();
  },
  train_live(message) {
    onLive(message);
  },
  measure_state(message) {
    app.measure = message.state;
    renderMeasure();
  },
  measure_live(message) {
    onMeasureLive(message);
  },
};

function connectEvents() {
  const source = new EventSource("/api/events");
  source.onmessage = (event) => {
    const message = JSON.parse(event.data);
    handlers[message.type]?.(message);
  };
  source.onopen = async () => {
    // Après une reconnexion, on resynchronise tout.
    try {
      renderGame(await api("/api/game"));
      await loadTraining();
      await loadMeasure();
    } catch (error) {
      toast(error.message);
    }
  };
}

/* ═══════════════════════════ Contrôles ═══════════════════════════ */

function setView(view) {
  app.view = view;
  $$(".tab").forEach((tab) => tab.classList.toggle("is-active", tab.dataset.view === view));
  for (const name of ["play", "train", "measure"]) $(`#view-${name}`).hidden = view !== name;
  if (view === "train") requestAnimationFrame(renderTraining);
  if (view === "measure") requestAnimationFrame(renderMeasure);
  try { localStorage.setItem("chessengine.view", view); } catch (_) { /* stockage indisponible */ }
}

function bindControls() {
  $$(".tab").forEach((tab) => tab.addEventListener("click", () => setView(tab.dataset.view)));

  $$("#color-choice button").forEach((button) => {
    button.addEventListener("click", () => {
      app.color = button.dataset.color;
      $$("#color-choice button").forEach((b) => b.classList.toggle("is-active", b === button));
    });
  });

  const think = $("#think");
  let thinkTimer = null;
  const showThink = () => {
    const ms = THINK_STEPS[app.thinkIndex];
    $("#think-out").textContent = ms < 1000 ? `${fmtDecimal(ms / 1000, 1)} s` : `${fmtDecimal(ms / 1000, ms % 1000 ? 1 : 0)} s`;
  };
  think.addEventListener("input", () => {
    app.thinkIndex = Number(think.value);
    showThink();
    clearTimeout(thinkTimer);
    thinkTimer = setTimeout(() => api("/api/game/settings", { think_ms: THINK_STEPS[app.thinkIndex] }).catch(() => {}), 250);
  });
  showThink();

  $("#new-game").addEventListener("click", () => newGame().catch((e) => toast(e.message)));
  $("#result-again").addEventListener("click", () => newGame().catch((e) => toast(e.message)));
  $("#undo").addEventListener("click", async () => {
    try {
      renderGame(await api("/api/game/undo", {}));
    } catch (error) {
      toast(error.message);
    }
  });
  $("#flip").addEventListener("click", () => {
    app.orientation = app.orientation === "w" ? "b" : "w";
    board.setOrientation(app.orientation);
    if (app.game) {
      renderPlayers(app.game);
      setEvalBar(app.game.engine);
    }
  });

  $("#train-toggle").addEventListener("click", async () => {
    try {
      if (app.train && app.train.running) {
        await api("/api/train/stop", {});
      } else {
        await api("/api/train/start", {
          games_per_generation: Number($("#cfg-games").value),
          nodes_per_move: Number($("#cfg-nodes").value),
        });
      }
    } catch (error) {
      toast(error.message);
    }
  });

  const menu = $("#reset-menu");
  $("#reset-toggle").addEventListener("click", (e) => {
    e.stopPropagation();
    menu.hidden = !menu.hidden;
  });
  document.addEventListener("click", (e) => { if (!menu.contains(e.target)) menu.hidden = true; });
  $$("#reset-menu button").forEach((button) => {
    button.addEventListener("click", async () => {
      menu.hidden = true;
      const origin = button.dataset.origin;
      const question = origin === "nnue"
        ? "Passer au réseau de neurones ? Ton champion actuel et l’historique sont conservés : un réseau va apprendre à battre ce champion."
        : `Effacer tout l’apprentissage et repartir ${origin === "zero" ? "de zéro" : "des valeurs classiques"} ?`;
      if (!confirm(question)) return;
      try {
        await api("/api/train/reset", { origin: button.dataset.origin });
        await loadTraining();
      } catch (error) {
        toast(error.message);
      }
    });
  });

  $$("#heat-phase button").forEach((button) => {
    button.addEventListener("click", () => {
      app.heatPhase = button.dataset.phase;
      $$("#heat-phase button").forEach((b) => b.classList.toggle("is-active", b === button));
      renderWeights();
    });
  });

  $("#m-toggle").addEventListener("click", async () => {
    try {
      if (app.measure && app.measure.running) {
        await api("/api/measure/stop", {});
      } else {
        const [base, increment] = $("#m-tc").value.split("/").map(Number);
        await api("/api/measure/start", {
          opponent_elo: Number($("#m-elo").value),
          games: Number($("#m-count").value),
          base_ms: base,
          increment_ms: increment,
        });
      }
    } catch (error) {
      toast(error.message);
    }
  });
  $("#sf-detect").addEventListener("click", async () => {
    try {
      app.measure = await api("/api/measure/detect", { path: $("#sf-path").value || null });
      renderMeasure();
    } catch (error) {
      toast(error.message);
    }
  });

  let resizeTimer = null;
  window.addEventListener("resize", () => {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(() => {
      if (app.view === "measure" && app.measure) drawMeasureChart();
      if (app.view === "train" && app.train) {
        drawEloChart();
        drawScoreChart();
        drawLossChart();
      }
    }, 120);
  });
}

async function start() {
  bindControls();
  renderHeatPieces();
  liveBoard.setPosition("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
  measureBoard.setPosition("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
  try {
    const game = await api("/api/game");
    app.orientation = game.human;
    board.setOrientation(app.orientation);
    app.color = game.human === "w" ? "white" : "black";
    $$("#color-choice button").forEach((b) => b.classList.toggle("is-active", b.dataset.color === app.color));
    const index = THINK_STEPS.indexOf(game.think_ms);
    if (index >= 0) {
      app.thinkIndex = index;
      $("#think").value = String(index);
      $("#think").dispatchEvent(new Event("input"));
    }
    renderGame(game);
    await loadTraining();
    await loadMeasure();
  } catch (error) {
    toast("Impossible de joindre le moteur : " + error.message);
  }
  let saved = null;
  try { saved = localStorage.getItem("chessengine.view"); } catch (_) { /* stockage indisponible */ }
  setView(["train", "measure"].includes(saved) ? saved : "play");
  connectEvents();
}

start();
