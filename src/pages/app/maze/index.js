import React, { useEffect, useState, useRef } from "react";
import Heading from "@theme/Heading";
import CitationNotice from "../../../components/CitationNotice";
import AppScaffold from "../../../components/AppScaffold";
import styles from "./styles.module.css";

const RANKING_KEY = "mazeRankings";
const MAX_RANKINGS = 10;
const MAX_TRAIL_LENGTH = 500;

function currentTimestamp() {
  return Date.now();
}

function readRankings() {
  try {
    const parsed = JSON.parse(localStorage.getItem(RANKING_KEY) || "[]");
    if (!Array.isArray(parsed)) return [];
    return parsed
      .filter(
        (item) =>
          item &&
          typeof item.name === "string" &&
          Number.isFinite(Number(item.time)) &&
          Number.isFinite(Number(item.steps))
      )
      .slice(0, MAX_RANKINGS)
      .map((item) => ({
        name: item.name.slice(0, 32),
        time: Number(item.time),
        steps: Number(item.steps),
      }));
  } catch {
    return [];
  }
}

function writeRankings(rankings) {
  try {
    localStorage.setItem(RANKING_KEY, JSON.stringify(rankings));
  } catch {
    // The game remains playable when storage is unavailable or full.
  }
}

export default function MazePage() {
  const [stepCount, setStepCount] = useState(0);
  const [ranking, setRanking] = useState([]);
  const canvasRef = useRef(null);
  const [playerName, setPlayerName] = useState("Player");
  const [message, setMessage] = useState("");
  const playerNameRef = useRef(playerName);
  const drawRef = useRef(null);
  const initGameRef = useRef(null);
  const moveBallRef = useRef(null);
  const imageAssetsRef = useRef({ hamburger: null, einstein: null });

  // Game state refs to avoid closure staleness in event listeners
  const gameState = useRef({
    maze: [],
    ball: { x: 0, y: 0, r: 0, c: 0 }, // r,c are grid coordinates
    cellSize: 0,
    rows: 21, // Odd number for better walls
    cols: 21,
    moves: 0,
    startTime: null,
    gameFinished: false,
    trail: [], // Array of {r, c}
  });

  useEffect(() => {
    playerNameRef.current = playerName;
  }, [playerName]);

  function initGame() {
    const canvas = canvasRef.current;
    if (!canvas) return;
    // Reset state
    gameState.current.moves = 0;
    gameState.current.startTime = null;
    gameState.current.gameFinished = false;
    gameState.current.trail = [];
    setStepCount(0);
    setMessage("");

    // Maze Dimensions
    // Ensure odd dimensions for proper wall generation
    const rows = 21;
    const cols = 21;
    gameState.current.rows = rows;
    gameState.current.cols = cols;

    // Calculate cell size based on current canvas width
    const cellSize = Math.floor(canvas.width / cols);
    gameState.current.cellSize = cellSize;

    // Generate Maze using Recursive Backtracker
    const maze = generateMazeRecursive(rows, cols);
    gameState.current.maze = maze;

    // Set Ball Position (Start at top-left 1,1)
    gameState.current.ball = {
      r: 1,
      c: 1,
      x: cellSize * 1.5,
      y: cellSize * 1.5,
      radius: cellSize * 0.35,
    };

    // Draw initial frame
    draw();
  }

  function generateMazeRecursive(rows, cols) {
    // Initialize full walls (1)
    let maze = Array(rows)
      .fill()
      .map(() => Array(cols).fill(1));

    // Carve from (1,1)
    function carve(r, c) {
      maze[r][c] = 0; // 0 is path

      // Randomize directions: Up, Right, Down, Left
      const dirs = [
        [-2, 0],
        [0, 2],
        [2, 0],
        [0, -2],
      ].sort(() => Math.random() - 0.5);

      for (let [dr, dc] of dirs) {
        const nr = r + dr;
        const nc = c + dc;
        if (
          nr > 0 &&
          nr < rows - 1 &&
          nc > 0 &&
          nc < cols - 1 &&
          maze[nr][nc] === 1
        ) {
          maze[r + dr / 2][c + dc / 2] = 0; // Carve wall between
          carve(nr, nc);
        }
      }
    }

    carve(1, 1);

    // Set Exit
    maze[rows - 2][cols - 1] = 2; // 2 is exit
    // Ensure path to exit
    if (maze[rows - 2][cols - 2] === 1) {
      maze[rows - 2][cols - 2] = 0;
    }

    return maze;
  }

  function draw() {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    const { maze, cellSize, ball, trail } = gameState.current;
    const rows = maze.length;
    const cols = maze[0].length;

    // Clear
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Draw Maze with images preloaded once for the lifetime of the page.
    const { hamburger: hamburgerImg, einstein: einsteinImg } =
      imageAssetsRef.current;

    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        if (maze[r][c] === 1) {
          ctx.fillStyle = "#2d3748";
          ctx.fillRect(c * cellSize, r * cellSize, cellSize, cellSize);
        } else if (maze[r][c] === 2) {
          // Draw Hamburger image for exit
          if (hamburgerImg?.complete && hamburgerImg.naturalWidth > 0) {
            ctx.drawImage(
              hamburgerImg,
              c * cellSize,
              r * cellSize,
              cellSize,
              cellSize
            );
          } else {
            ctx.fillStyle = "#48bb78";
            ctx.fillRect(c * cellSize, r * cellSize, cellSize, cellSize);
          }
        } else {
          ctx.fillStyle = "#ffffff";
          ctx.fillRect(c * cellSize, r * cellSize, cellSize, cellSize);
        }
      }
    }

    // Draw Trail
    ctx.fillStyle = "#feb2b2"; // Trail color
    for (let t of trail) {
      ctx.fillRect(t.c * cellSize, t.r * cellSize, cellSize, cellSize);
    }

    // Draw Ball (Einstein)
    const ballX = ball.c * cellSize + cellSize / 2;
    const ballY = ball.r * cellSize + cellSize / 2;
    const ballRadius = ball.radius;

    if (einsteinImg?.complete && einsteinImg.naturalWidth > 0) {
      ctx.save();
      ctx.beginPath();
      ctx.arc(ballX, ballY, ballRadius, 0, Math.PI * 2);
      ctx.clip();
      ctx.drawImage(
        einsteinImg,
        ballX - ballRadius,
        ballY - ballRadius,
        ballRadius * 2,
        ballRadius * 2
      );
      ctx.restore();

      // Add border
      ctx.beginPath();
      ctx.strokeStyle = "#fff";
      ctx.lineWidth = 2;
      ctx.arc(ballX, ballY, ballRadius, 0, Math.PI * 2);
      ctx.stroke();
    } else {
      ctx.fillStyle = "#f5c542";
      ctx.beginPath();
      ctx.arc(ballX, ballY, ballRadius, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  function moveBall(dr, dc) {
    if (gameState.current.gameFinished) return;

    if (!gameState.current.startTime) {
      gameState.current.startTime = currentTimestamp();
    }

    const { maze, ball } = gameState.current;
    const newR = ball.r + dr;
    const newC = ball.c + dc;

    // Check bounds and walls
    if (maze[newR] && maze[newR][newC] !== 1) {
      // Valid move

      // Add previous pos to trail only if not backtracking (simple trail)
      gameState.current.trail.push({ r: ball.r, c: ball.c });
      if (gameState.current.trail.length > MAX_TRAIL_LENGTH) {
        gameState.current.trail.shift();
      }

      gameState.current.ball.r = newR;
      gameState.current.ball.c = newC;
      gameState.current.moves++;

      setStepCount(gameState.current.moves);

      checkExit();
      draw();
    }
  }

  function checkExit() {
    const { maze, ball, moves, startTime } = gameState.current;
    if (maze[ball.r][ball.c] === 2) {
      gameState.current.gameFinished = true;
      const endTime = currentTimestamp();
      const elapsedSeconds = ((endTime - startTime) / 1000).toFixed(2);

      const safeName = playerNameRef.current.trim().slice(0, 32) || "Player";
      setMessage(
        `Victory! ${safeName} finished in ${elapsedSeconds}s with ${moves} moves!`
      );
      recordRanking(safeName, parseFloat(elapsedSeconds), moves);
    }
  }

  function recordRanking(name, time, steps) {
    let rankings = readRankings();
    rankings.push({ name, time, steps });
    // Sort by time, then steps
    rankings.sort((a, b) => a.time - b.time || a.steps - b.steps);
    // Keep top 10
    rankings = rankings.slice(0, MAX_RANKINGS);
    writeRankings(rankings);
    setRanking(rankings);
  }

  useEffect(() => {
    drawRef.current = draw;
    initGameRef.current = initGame;
    moveBallRef.current = moveBall;
  });

  useEffect(() => {
    const rankingFrame = window.requestAnimationFrame(() => {
      setRanking(readRankings());
    });

    const hamburger = new Image();
    const einstein = new Image();
    imageAssetsRef.current = { hamburger, einstein };
    const redraw = () => drawRef.current?.();
    hamburger.addEventListener("load", redraw);
    einstein.addEventListener("load", redraw);
    hamburger.src = "/img/Hamburger.png";
    einstein.src = "/img/Einstein.png";

    initGameRef.current?.();

    const onKeyDown = (e) => {
      const target = e.target;
      if (
        target instanceof HTMLElement &&
        (target.isContentEditable ||
          ["INPUT", "TEXTAREA", "SELECT", "BUTTON"].includes(target.tagName))
      ) {
        return;
      }
      if (
        ["ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight"].includes(e.code)
      ) {
        e.preventDefault();
      }
      switch (e.key) {
        case "ArrowUp":
          moveBallRef.current?.(-1, 0);
          break;
        case "ArrowDown":
          moveBallRef.current?.(1, 0);
          break;
        case "ArrowLeft":
          moveBallRef.current?.(0, -1);
          break;
        case "ArrowRight":
          moveBallRef.current?.(0, 1);
          break;
      }
    };
    window.addEventListener("keydown", onKeyDown);

    return () => {
      window.cancelAnimationFrame(rankingFrame);
      window.removeEventListener("keydown", onKeyDown);
      hamburger.removeEventListener("load", redraw);
      einstein.removeEventListener("load", redraw);
      imageAssetsRef.current = { hamburger: null, einstein: null };
    };
  }, []);

  return (
    <AppScaffold appId="maze">
      <div className={styles.layout}>
        <section className={styles.gameCard} aria-labelledby="maze-board-title">
          <div className={styles.toolbar}>
            <div className={styles.playerField}>
              <label htmlFor="maze-player">Player</label>
              <input
                id="maze-player"
                type="text"
                value={playerName}
                onChange={(e) => setPlayerName(e.target.value)}
                maxLength={32}
                autoComplete="nickname"
              />
            </div>
            <div className={styles.steps} aria-live="polite">
              <span>Steps</span>
              <strong>{stepCount}</strong>
            </div>
            <button
              type="button"
              onClick={initGame}
              className={styles.newGameButton}
            >
              New Game
            </button>
          </div>

          <div className={styles.boardHeading}>
            <div>
              <span>Procedural board</span>
              <Heading as="h2" id="maze-board-title">
                Find the green exit
              </Heading>
            </div>
            <span className={styles.keyboardHint}>Arrow keys or controls</span>
          </div>

          <div className={styles.canvasFrame}>
            <canvas
              ref={canvasRef}
              id="mazeCanvas"
              width="840"
              height="840"
              className={styles.canvas}
              role="img"
              aria-label={`21 by 21 maze board. ${stepCount} moves completed.${
                message ? ` ${message}` : ""
              }`}
            >
              A 21 by 21 maze. Use the arrow controls to move from the top-left
              to the green exit.
            </canvas>
            <div className={styles.message} role="status" aria-live="assertive">
              {message}
            </div>
          </div>

          <div className={styles.dpad} aria-label="Maze direction controls">
            <span />
            <button
              type="button"
              onClick={() => moveBall(-1, 0)}
              aria-label="Move up"
            >
              ▲
            </button>
            <span />

            <button
              type="button"
              onClick={() => moveBall(0, -1)}
              aria-label="Move left"
            >
              ◀
            </button>

            <button
              type="button"
              onClick={() => moveBall(1, 0)}
              aria-label="Move down"
            >
              ▼
            </button>

            <button
              type="button"
              onClick={() => moveBall(0, 1)}
              aria-label="Move right"
            >
              ▶
            </button>
          </div>
        </section>

        <aside
          className={styles.leaderboard}
          aria-labelledby="maze-leaderboard-title"
        >
          <div className={styles.leaderboardHeading}>
            <span>Local records</span>
            <Heading as="h2" id="maze-leaderboard-title">
              Leaderboard
            </Heading>
          </div>
          <ol>
            {ranking.length === 0 && (
              <li className={styles.emptyRanking}>
                Finish a maze to set the first record.
              </li>
            )}
            {ranking.map((r, i) => (
              <li key={`${r.name}-${r.time}-${r.steps}-${i}`}>
                <span className={styles.place}>{i + 1}</span>
                <span>
                  <strong>{r.name}</strong>
                  <small>{r.steps} steps</small>
                </span>
                <strong>{r.time}s</strong>
              </li>
            ))}
          </ol>
          <p>Scores stay in this browser and are never uploaded.</p>
        </aside>
      </div>

      <CitationNotice />
    </AppScaffold>
  );
}
