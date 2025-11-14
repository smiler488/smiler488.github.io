import React, { useEffect, useState } from "react";
import Layout from "@theme/Layout";
import Head from "@docusaurus/Head";
import CitationNotice from "../../../components/CitationNotice";

export default function MazePage() {
  // 使用 React 状态管理步数和排名显示（也可直接操作 DOM）
  const [stepCount, setStepCount] = useState(0);
  const [ranking, setRanking] = useState([]);

  useEffect(() => {
    // 获取玩家名称输入框和其他 DOM 元素
    let playerName = "Player"; // 默认名称
    const nameInput = document.getElementById("playerName");
    nameInput.addEventListener("change", (e) => {
      playerName = e.target.value.trim() || "Player";
    });

    // 获取画布和上下文
    const canvas = document.getElementById("mazeCanvas");
    const ctx = canvas.getContext("2d");

    const css = getComputedStyle(document.documentElement);
    const getVar = (name, fallback) => (css.getPropertyValue(name).trim() || fallback);

    // --- 随机生成迷宫 ---
    const rows = 20;
    const cols = 20;
    const cellSize = Math.floor(canvas.width / cols);

    function generateMaze(rows, cols) {
      const maze = [];
      for (let i = 0; i < rows; i++) {
        maze[i] = [];
        for (let j = 0; j < cols; j++) {
          if (i === 0 || j === 0 || i === rows - 1 || j === cols - 1) {
            maze[i][j] = 1; // 墙
          } else {
            // 内部随机：70% 概率空，30% 概率墙
            maze[i][j] = Math.random() < 0.7 ? 0 : 1;
          }
        }
      }
      // 固定起点（1,1）为空
      maze[1][1] = 0;
      // 出口：右边界随机一个非角落单元设为 2
      const exitRow = Math.floor(Math.random() * (rows - 2)) + 1;
      maze[exitRow][cols - 1] = 2;
      return maze;
    }
    let maze = generateMaze(rows, cols);

    // --- 全局变量 ---
    let moves = 0;          // 移动步数
    let startTime = null;   // 计时开始时刻
    let gameFinished = false;

    // 小球初始状态，从起点 (1,1) 的中心出发
    let ball = {
      x: cellSize * 1 + cellSize / 2,
      y: cellSize * 1 + cellSize / 2,
      radius: cellSize / 4,
      speed: cellSize, // 每次移动一个格子
    };

    // --- 绘制迷宫 ---
    function drawMaze() {
      for (let i = 0; i < maze.length; i++) {
        for (let j = 0; j < maze[i].length; j++) {
          if (maze[i][j] === 1) {
            ctx.fillStyle = getVar('--ifm-color-emphasis-800', '#333');
          } else if (maze[i][j] === 2) {
            ctx.fillStyle = getVar('--ifm-color-emphasis-600', '#999999');
          } else if (maze[i][j] === 3) {
            ctx.fillStyle = getVar('--app-accent-muted', '#e5e5ea');
          } else {
            ctx.fillStyle = getVar('--ifm-background-color', '#ffffff');
          }
          ctx.fillRect(j * cellSize, i * cellSize, cellSize, cellSize);
          ctx.strokeStyle = getVar('--ifm-border-color', '#cccccc');
          ctx.strokeRect(j * cellSize, i * cellSize, cellSize, cellSize);
        }
      }
    }

    // --- 绘制小球 ---
    function drawBall() {
      ctx.beginPath();
      ctx.fillStyle = getVar('--ifm-color-emphasis-900', '#000000');
      ctx.arc(ball.x, ball.y, ball.radius, 0, Math.PI * 2);
      ctx.fill();
    }

    // --- 更新步数显示 ---
    function updateStepCounter() {
      setStepCount(moves);
      const counterElem = document.getElementById("stepCounter");
      if (counterElem) {
        counterElem.innerText = moves;
      }
    }

    // --- 主动画循环 ---
    function animate() {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      drawMaze();
      drawBall();
      checkExit();
      requestAnimationFrame(animate);
    }
    animate();

    // --- 小球移动函数 --- 
    // 按一次按键移动一格，同时标记目标格为粉色
    function moveBall(dx, dy) {
      if (!startTime) startTime = Date.now(); // 第一次移动时开始计时

      const newX = ball.x + dx;
      const newY = ball.y + dy;
      const col = Math.floor(newX / cellSize);
      const row = Math.floor(newY / cellSize);

      // 如果遇到墙体，直接返回不移动
      if (maze[row][col] === 1) return;

      // 更新小球位置
      ball.x = newX;
      ball.y = newY;
      moves++;
      updateStepCounter();

      // 涂鸦：如果该格是空格 0，则标记为粉色（3）
      if (maze[row][col] === 0) {
        maze[row][col] = 3;
      }
    }

    // --- 按钮控制 ---
    document.getElementById("btnUp").addEventListener("click", () => moveBall(0, -ball.speed));
    document.getElementById("btnDown").addEventListener("click", () => moveBall(0, ball.speed));
    document.getElementById("btnLeft").addEventListener("click", () => moveBall(-ball.speed, 0));
    document.getElementById("btnRight").addEventListener("click", () => moveBall(ball.speed, 0));

    // --- 键盘控制（箭头键） ---
    window.addEventListener("keydown", (e) => {
      switch (e.key) {
        case "ArrowUp":
          moveBall(0, -ball.speed);
          break;
        case "ArrowDown":
          moveBall(0, ball.speed);
          break;
        case "ArrowLeft":
          moveBall(-ball.speed, 0);
          break;
        case "ArrowRight":
          moveBall(ball.speed, 0);
          break;
        default:
          break;
      }
    });

    // --- 检查是否到达出口 ---
    function checkExit() {
      const col = Math.floor(ball.x / cellSize);
      const row = Math.floor(ball.y / cellSize);
      if (maze[row][col] === 2 && !gameFinished) {
        gameFinished = true;
        const endTime = Date.now();
        const elapsedSeconds = Math.floor((endTime - startTime) / 1000);
        document.getElementById("mazeMessage").innerText =
          `Congratulations, ${playerName || "Player"}! You've reached the exit in ${elapsedSeconds} seconds and ${moves} moves!`;
        recordRanking(playerName || "Player", elapsedSeconds, moves);
      } else if (maze[row][col] !== 2) {
        document.getElementById("mazeMessage").innerText = "";
      }
    }

    // --- 记录排名到 localStorage 并更新排名显示 ---
    function recordRanking(name, time, steps) {
      let rankings = JSON.parse(localStorage.getItem("mazeRankings")) || [];
      rankings.push({ name, time, steps });
      rankings.sort((a, b) => a.time - b.time || a.steps - b.steps);
      localStorage.setItem("mazeRankings", JSON.stringify(rankings));
      setRanking(rankings);
      updateRankingDisplay();
    }

    // --- 更新排名显示 ---
    function updateRankingDisplay() {
      const rankingDiv = document.getElementById("ranking");
      let html = "<h3>Ranking</h3><ol>";
      const rankings = JSON.parse(localStorage.getItem("mazeRankings")) || [];
      rankings.forEach((entry) => {
        html += `<li>${entry.name} - ${entry.time} sec, ${entry.steps} moves</li>`;
      });
      html += "</ol>";
      rankingDiv.innerHTML = html;
    }

    // --- 简单的 React 状态模拟：更新排名（这里直接操作 localStorage 后更新显示） ---
    function setRanking(rankings) {
      setRanking(rankings); // 如果需要，可以使用 React 状态管理，但这里我们直接操作 DOM
    }

    // --- 重置按钮：重置小球位置、步数和涂鸦 ---
    document.getElementById("resetStepsBtn").addEventListener("click", () => {
      moves = 0;
      updateStepCounter();
      // 重置小球位置到起点
      ball.x = cellSize * 1 + cellSize / 2;
      ball.y = cellSize * 1 + cellSize / 2;
      // 清除涂鸦：将 maze 中所有标记为 3 的格子恢复为空格 0
      for (let i = 0; i < maze.length; i++) {
        for (let j = 0; j < maze[i].length; j++) {
          if (maze[i][j] === 3) {
            maze[i][j] = 0;
          }
        }
      }
    });

    // --- 初始化排名显示 ---
    const storedRankings = JSON.parse(localStorage.getItem("mazeRankings"));
    if (storedRankings) {
      updateRankingDisplay();
    }
  }, []);

  return (
    <Layout title="2D Marble Maze">
      <Head>
        <meta name="viewport" content="width=device-width, initial-scale=1.0" />
      </Head>
      <div className="app-container">
        <div className="app-header" style={{ marginBottom: 16 }}>
          <h1 className="app-title">2D Marble Maze</h1>
          <a className="button button--secondary" href="/docs/tutorial-apps/maze-game-tutorial">Tutorial</a>
        </div>
        <div className="app-card" style={{ marginBottom: 12, display: typeof window !== 'undefined' && !window.__APP_AUTH_OK__ ? 'block' : 'none' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <span className="app-muted">Please login to use app features</span>
            <a className="button button--secondary" href="/auth">Login / Register</a>
          </div>
        </div>
        <div style={{ marginBottom: "10px" }}>
          <label htmlFor="playerName">Player Name: </label>
          <input
            type="text"
            id="playerName"
            placeholder="Enter your name"
            style={{ padding: "5px", borderRadius: "5px", border: "1px solid var(--ifm-border-color)" }}
          />
        </div>
        <div style={{ marginBottom: "10px", fontSize: "1.2rem" }}>
          Steps: <span id="stepCounter">0</span>
        </div>
        <canvas id="mazeCanvas" width="800" height="800" style={{ border: "2px solid var(--ifm-color-emphasis-800)" }}></canvas>
        <div id="mazeMessage" style={{ marginTop: "10px", fontSize: "1.2rem" }}></div>
        <div style={{ marginTop: "20px" }}>
          <button id="btnUp" style={{ padding: "10px 15px", margin: "5px" }}>↑</button>
          <div>
            <button id="btnLeft" style={{ padding: "10px 15px", margin: "5px" }}>←</button>
            <button id="btnDown" style={{ padding: "10px 15px", margin: "5px" }}>↓</button>
            <button id="btnRight" style={{ padding: "10px 15px", margin: "5px" }}>→</button>
          </div>
          <button id="resetStepsBtn" style={{ padding: "10px 15px", margin: "5px" }}>
             Reset Steps & Doodle
          </button>
        </div>
        <div id="mazeMessage" style={{ marginTop: "10px", fontSize: "1.2rem" }}></div>
        <div id="ranking" style={{ marginTop: "20px" }}></div>
        <CitationNotice />
      </div>
    </Layout>
  );
}
