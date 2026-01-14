import React, { useEffect, useState, useRef } from "react";
import Layout from "@theme/Layout";
import Head from "@docusaurus/Head";
import CitationNotice from "../../../components/CitationNotice";

export default function MazePage() {
  const [stepCount, setStepCount] = useState(0);
  const [ranking, setRanking] = useState([]);
  const canvasRef = useRef(null);
  const [playerName, setPlayerName] = useState("Player");

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
    trail: [] // Array of {r, c}
  });

  useEffect(() => {
    // Load ranking
    const storedRankings = JSON.parse(localStorage.getItem("mazeRankings")) || [];
    setRanking(storedRankings);

    initGame();

    // keydown listener
    const onKeyDown = (e) => {
      // Prevent scrolling if arrow keys
      if (["ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight"].indexOf(e.code) > -1) {
        e.preventDefault();
      }
      switch (e.key) {
        case "ArrowUp": moveBall(-1, 0); break;
        case "ArrowDown": moveBall(1, 0); break;
        case "ArrowLeft": moveBall(0, -1); break;
        case "ArrowRight": moveBall(0, 1); break;
      }
    };
    window.addEventListener("keydown", onKeyDown);

    return () => {
      window.removeEventListener("keydown", onKeyDown);
    };
  }, []);

  function initGame() {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");

    // Reset state
    gameState.current.moves = 0;
    gameState.current.startTime = null;
    gameState.current.gameFinished = false;
    gameState.current.trail = [];
    setStepCount(0);
    const msg = document.getElementById("mazeMessage");
    if (msg) msg.innerText = "";

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
      radius: cellSize * 0.35
    };

    // Draw initial frame
    draw();
  }

  function generateMazeRecursive(rows, cols) {
    // Initialize full walls (1)
    let maze = Array(rows).fill().map(() => Array(cols).fill(1));

    // Carve from (1,1)
    function carve(r, c) {
      maze[r][c] = 0; // 0 is path

      // Randomize directions: Up, Right, Down, Left
      const dirs = [
        [-2, 0], [0, 2], [2, 0], [0, -2]
      ].sort(() => Math.random() - 0.5);

      for (let [dr, dc] of dirs) {
        const nr = r + dr;
        const nc = c + dc;
        if (nr > 0 && nr < rows - 1 && nc > 0 && nc < cols - 1 && maze[nr][nc] === 1) {
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

    // Draw Maze
    const hamburgerImg = new Image();
    hamburgerImg.src = "/img/Hamburger.png";
    
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        if (maze[r][c] === 1) {
          ctx.fillStyle = "#2d3748";
          ctx.fillRect(c * cellSize, r * cellSize, cellSize, cellSize);
        } else if (maze[r][c] === 2) {
          // Draw Hamburger image for exit
          const drawHamburger = () => {
            ctx.drawImage(
              hamburgerImg,
              c * cellSize,
              r * cellSize,
              cellSize,
              cellSize
            );
          };
          
          if (hamburgerImg.complete) {
            drawHamburger();
          } else {
            hamburgerImg.onload = drawHamburger;
            // Fallback while loading
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
    
    // Load and draw Einstein image
    const einsteinImg = new Image();
    einsteinImg.src = "/img/Einstein.png";
    einsteinImg.onload = () => {
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
    };
    
    // Fallback if image doesn't load immediately
    if (einsteinImg.complete) {
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
      
      ctx.beginPath();
      ctx.strokeStyle = "#fff";
      ctx.lineWidth = 2;
      ctx.arc(ballX, ballY, ballRadius, 0, Math.PI * 2);
      ctx.stroke();
    }
  }

  function moveBall(dr, dc) {
    if (gameState.current.gameFinished) return;

    if (!gameState.current.startTime) {
      gameState.current.startTime = Date.now();
    }

    const { maze, ball } = gameState.current;
    const newR = ball.r + dr;
    const newC = ball.c + dc;

    // Check bounds and walls
    if (maze[newR] && maze[newR][newC] !== 1) {
      // Valid move

      // Add previous pos to trail only if not backtracking (simple trail)
      gameState.current.trail.push({ r: ball.r, c: ball.c });

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
      const endTime = Date.now();
      const elapsedSeconds = ((endTime - startTime) / 1000).toFixed(2);

      const msg = document.getElementById("mazeMessage");
      if (msg) {
        msg.innerText = `Victory! ${playerName} finished in ${elapsedSeconds}s with ${moves} moves!`;
      }

      recordRanking(playerName, parseFloat(elapsedSeconds), moves);
    }
  }

  function recordRanking(name, time, steps) {
    let rankings = JSON.parse(localStorage.getItem("mazeRankings")) || [];
    rankings.push({ name, time, steps });
    // Sort by time, then steps
    rankings.sort((a, b) => a.time - b.time || a.steps - b.steps);
    // Keep top 10
    rankings = rankings.slice(0, 10);
    localStorage.setItem("mazeRankings", JSON.stringify(rankings));
    setRanking(rankings);
  }

  // Prevent default scroll when touching control buttons
  const preventScroll = (e) => {
    if (e.cancelable) e.preventDefault();
  };

  return (
    <Layout title="2D Marble Maze">
      <Head>
        <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no" />
      </Head>
      <div className="app-container" style={{ maxWidth: '850px', margin: '0 auto', padding: '1rem', paddingBottom: '3rem' }}>
        <div className="app-header" style={{ marginBottom: 16, textAlign: 'center' }}>
          <h1 className="app-title" style={{ fontSize: '2rem', marginBottom: '0.5rem' }}>Marble Maze</h1>
          <p style={{ color: 'var(--ifm-color-emphasis-600)' }}>Navigate to the green exit</p>
        </div>

        <div className="app-card" style={{ padding: '0.8rem', marginBottom: '1rem', display: 'flex', flexDirection: 'column', gap: '0.8rem' }}>
          <div style={{ display: "flex", justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '8px' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
              <label style={{ fontWeight: 'bold', fontSize: '0.9rem' }}>Player:</label>
              <input
                type="text"
                value={playerName}
                onChange={(e) => setPlayerName(e.target.value)}
                style={{ width: '100px', padding: "4px 8px", borderRadius: "6px", border: "1px solid var(--ifm-border-color)" }}
              />
            </div>
            <div style={{ fontSize: "1rem", fontWeight: 'bold' }}>
              Steps: <span style={{ color: 'var(--ifm-color-primary)' }}>{stepCount}</span>
            </div>
            <button
              onClick={initGame}
              className="button button--primary button--sm"
            >
              New Game
            </button>
          </div>

          {/* Flexible container for canvas to fit screen */}
          <div style={{
            position: 'relative',
            width: '100%',
            maxWidth: '500px',
            margin: '0 auto',
            aspectRatio: '1/1'
          }}>
            <canvas
              ref={canvasRef}
              id="mazeCanvas"
              width="840"
              height="840"
              style={{
                width: '100%',
                height: '100%',
                borderRadius: "8px",
                background: '#2d3748',
                touchAction: 'none', // Prevent scrolling on touch
                display: 'block'
              }}
            ></canvas>
            <div id="mazeMessage" style={{
              position: 'absolute',
              bottom: '50%',
              left: 0,
              width: '100%',
              textAlign: 'center',
              textShadow: '0 2px 4px rgba(0,0,0,0.8)',
              color: '#fff',
              fontWeight: 'bold',
              fontSize: '1.2rem',
              pointerEvents: 'none'
            }}></div>
          </div>

          {/* D-Pad Controls for Touch/Mouse */}
          <div
            style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(3, 1fr)',
              gap: '8px',
              maxWidth: '160px',
              margin: '0 auto',
              touchAction: 'none'
            }}
          >
            <div></div>
            <button
              className="button button--secondary"
              onTouchStart={(e) => { preventScroll(e); moveBall(-1, 0); }}
              onClick={() => moveBall(-1, 0)}
              style={{ padding: '16px', fontSize: '1.2rem', lineHeight: 1 }}
            >▲</button>
            <div></div>

            <button
              className="button button--secondary"
              onTouchStart={(e) => { preventScroll(e); moveBall(0, -1); }}
              onClick={() => moveBall(0, -1)}
              style={{ padding: '16px', fontSize: '1.2rem', lineHeight: 1 }}
            >◀</button>

            <button
              className="button button--secondary"
              onTouchStart={(e) => { preventScroll(e); moveBall(1, 0); }}
              onClick={() => moveBall(1, 0)}
              style={{ padding: '16px', fontSize: '1.2rem', lineHeight: 1 }}
            >▼</button>

            <button
              className="button button--secondary"
              onTouchStart={(e) => { preventScroll(e); moveBall(0, 1); }}
              onClick={() => moveBall(0, 1)}
              style={{ padding: '16px', fontSize: '1.2rem', lineHeight: 1 }}
            >▶</button>
          </div>
          <div style={{ textAlign: 'center', fontSize: '0.8rem', color: 'var(--ifm-color-emphasis-500)' }}>
            Tap buttons or use arrow keys
          </div>
        </div>

        <div className="app-card" style={{ marginTop: "1rem" }}>
          <h3>Leaderboard</h3>
          <ol style={{ paddingLeft: '1.5rem', margin: 0 }}>
            {ranking.length === 0 && <li style={{ color: 'var(--ifm-color-emphasis-500)' }}>No records yet.</li>}
            {ranking.map((r, i) => (
              <li key={i} style={{ marginBottom: 4 }}>
                <strong>{r.name}</strong> — {r.time}s ({r.steps} steps)
              </li>
            ))}
          </ol>
        </div>

        <CitationNotice />
      </div>
    </Layout>
  );
}
