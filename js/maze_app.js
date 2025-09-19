/**
 * Enhanced Maze Game with Multiple Features
 * Features: Multiple game modes, power-ups, enemies, sound effects, achievements
 */

class EnhancedMazeGame {
    constructor() {
        this.canvas = null;
        this.ctx = null;
        this.gameState = 'menu'; // 'menu', 'playing', 'paused', 'gameOver', 'victory'
        this.gameMode = 'classic'; // 'classic', 'timed', 'survival', 'puzzle'
        this.difficulty = 'medium'; // 'easy', 'medium', 'hard', 'expert'
        
        // Game settings based on difficulty
        this.settings = {
            easy: { rows: 15, cols: 15, wallDensity: 0.25, timeLimit: 300 },
            medium: { rows: 20, cols: 20, wallDensity: 0.3, timeLimit: 240 },
            hard: { rows: 25, cols: 25, wallDensity: 0.35, timeLimit: 180 },
            expert: { rows: 30, cols: 30, wallDensity: 0.4, timeLimit: 120 }
        };
        
        // Game state
        this.maze = [];
        this.player = {};
        this.enemies = [];
        this.powerUps = [];
        this.particles = [];
        this.achievements = [];
        
        // Game stats
        this.moves = 0;
        this.startTime = null;
        this.gameTime = 0;
        this.score = 0;
        this.lives = 3;
        this.level = 1;
        
        // Player abilities
        this.playerAbilities = {
            speed: 1,
            wallBreaker: false,
            invisible: false,
            timeFreeze: false,
            shield: false
        };
        
        // Sound effects (using Web Audio API)
        this.sounds = {};
        this.soundEnabled = true;
        
        // Animation
        this.animationId = null;
        this.lastTime = 0;
        
        this.initializeGame();
    }

    initializeGame() {
        this.canvas = document.getElementById('mazeCanvas');
        this.ctx = this.canvas.getContext('2d');
        
        // Make canvas responsive
        this.resizeCanvas();
        window.addEventListener('resize', () => this.resizeCanvas());
        
        this.setupEventListeners();
        this.loadAchievements();
        this.initializeSounds();
        this.showMainMenu();
    }

    resizeCanvas() {
        const container = this.canvas.parentElement;
        const maxSize = Math.min(container.clientWidth - 40, 800);
        this.canvas.width = maxSize;
        this.canvas.height = maxSize;
    }

    setupEventListeners() {
        // Keyboard controls
        document.addEventListener('keydown', (e) => this.handleKeyPress(e));
        
        // Touch controls for mobile
        this.canvas.addEventListener('touchstart', (e) => this.handleTouch(e));
        this.canvas.addEventListener('touchmove', (e) => e.preventDefault());
        
        // Game control buttons
        this.setupGameButtons();
    }

    setupGameButtons() {
        // Create enhanced control panel
        const controlPanel = document.createElement('div');
        controlPanel.id = 'mazeControlPanel';
        controlPanel.innerHTML = `
            <div class="maze-menu" id="mazeMainMenu">
                <h2>🎮 Enhanced Maze Adventure</h2>
                <div class="menu-section">
                    <h3>Game Mode</h3>
                    <select id="gameModeSelect">
                        <option value="classic">Classic Mode</option>
                        <option value="timed">Time Challenge</option>
                        <option value="survival">Survival Mode</option>
                        <option value="puzzle">Puzzle Mode</option>
                    </select>
                </div>
                <div class="menu-section">
                    <h3>Difficulty</h3>
                    <select id="difficultySelect">
                        <option value="easy">Easy (15x15)</option>
                        <option value="medium" selected>Medium (20x20)</option>
                        <option value="hard">Hard (25x25)</option>
                        <option value="expert">Expert (30x30)</option>
                    </select>
                </div>
                <div class="menu-buttons">
                    <button id="startGameBtn" class="btn-primary">🚀 Start Game</button>
                    <button id="instructionsBtn" class="btn-secondary">📖 Instructions</button>
                    <button id="achievementsBtn" class="btn-secondary">🏆 Achievements</button>
                    <button id="settingsBtn" class="btn-secondary">⚙️ Settings</button>
                </div>
            </div>
            
            <div class="game-hud" id="gameHUD" style="display: none;">
                <div class="hud-top">
                    <div class="hud-item">
                        <span class="hud-label">Score:</span>
                        <span id="scoreDisplay">0</span>
                    </div>
                    <div class="hud-item">
                        <span class="hud-label">Time:</span>
                        <span id="timeDisplay">0:00</span>
                    </div>
                    <div class="hud-item">
                        <span class="hud-label">Lives:</span>
                        <span id="livesDisplay">❤️❤️❤️</span>
                    </div>
                    <div class="hud-item">
                        <span class="hud-label">Level:</span>
                        <span id="levelDisplay">1</span>
                    </div>
                </div>
                <div class="hud-bottom">
                    <div class="power-ups" id="activePowerUps"></div>
                    <div class="game-controls">
                        <button id="pauseBtn" class="btn-small">⏸️</button>
                        <button id="menuBtn" class="btn-small">🏠</button>
                    </div>
                </div>
            </div>
            
            <div class="mobile-controls" id="mobileControls" style="display: none;">
                <div class="control-pad">
                    <button class="control-btn" id="upBtn">↑</button>
                    <div class="control-row">
                        <button class="control-btn" id="leftBtn">←</button>
                        <button class="control-btn" id="downBtn">↓</button>
                        <button class="control-btn" id="rightBtn">→</button>
                    </div>
                </div>
                <div class="action-buttons">
                    <button class="action-btn" id="actionBtn">🔥 Action</button>
                </div>
            </div>
        `;
        
        // Insert control panel after canvas
        this.canvas.parentNode.insertBefore(controlPanel, this.canvas.nextSibling);
        
        // Setup button event listeners
        document.getElementById('startGameBtn').addEventListener('click', () => this.startNewGame());
        document.getElementById('pauseBtn').addEventListener('click', () => this.togglePause());
        document.getElementById('menuBtn').addEventListener('click', () => this.showMainMenu());
        
        // Mobile controls
        document.getElementById('upBtn').addEventListener('click', () => this.movePlayer(0, -1));
        document.getElementById('downBtn').addEventListener('click', () => this.movePlayer(0, 1));
        document.getElementById('leftBtn').addEventListener('click', () => this.movePlayer(-1, 0));
        document.getElementById('rightBtn').addEventListener('click', () => this.movePlayer(1, 0));
        document.getElementById('actionBtn').addEventListener('click', () => this.usePlayerAction());
        
        // Settings
        document.getElementById('gameModeSelect').addEventListener('change', (e) => {
            this.gameMode = e.target.value;
        });
        document.getElementById('difficultySelect').addEventListener('change', (e) => {
            this.difficulty = e.target.value;
        });
    }

    initializeSounds() {
        // Create simple sound effects using Web Audio API
        this.audioContext = new (window.AudioContext || window.webkitAudioContext)();
        
        this.sounds = {
            move: () => this.playTone(200, 0.1, 'sine'),
            collect: () => this.playTone(400, 0.2, 'sine'),
            victory: () => this.playMelody([262, 294, 330, 349], 0.3),
            defeat: () => this.playTone(150, 0.5, 'sawtooth'),
            powerUp: () => this.playMelody([400, 500, 600], 0.2),
            enemy: () => this.playTone(100, 0.3, 'triangle')
        };
    }

    playTone(frequency, duration, type = 'sine') {
        if (!this.soundEnabled) return;
        
        const oscillator = this.audioContext.createOscillator();
        const gainNode = this.audioContext.createGain();
        
        oscillator.connect(gainNode);
        gainNode.connect(this.audioContext.destination);
        
        oscillator.frequency.value = frequency;
        oscillator.type = type;
        
        gainNode.gain.setValueAtTime(0.1, this.audioContext.currentTime);
        gainNode.gain.exponentialRampToValueAtTime(0.01, this.audioContext.currentTime + duration);
        
        oscillator.start(this.audioContext.currentTime);
        oscillator.stop(this.audioContext.currentTime + duration);
    }

    playMelody(frequencies, noteDuration) {
        frequencies.forEach((freq, index) => {
            setTimeout(() => this.playTone(freq, noteDuration), index * noteDuration * 1000);
        });
    }

    showMainMenu() {
        this.gameState = 'menu';
        document.getElementById('mazeMainMenu').style.display = 'block';
        document.getElementById('gameHUD').style.display = 'none';
        document.getElementById('mobileControls').style.display = 'none';
        
        if (this.animationId) {
            cancelAnimationFrame(this.animationId);
        }
        
        this.drawMenuBackground();
    }

    drawMenuBackground() {
        // Draw animated background
        this.ctx.fillStyle = '#1a1a2e';
        this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
        
        // Draw animated particles
        const time = Date.now() * 0.001;
        for (let i = 0; i < 50; i++) {
            const x = (Math.sin(time + i) * 0.5 + 0.5) * this.canvas.width;
            const y = (Math.cos(time * 0.7 + i) * 0.5 + 0.5) * this.canvas.height;
            const size = Math.sin(time * 2 + i) * 2 + 3;
            
            this.ctx.fillStyle = `hsl(${(time * 50 + i * 10) % 360}, 70%, 60%)`;
            this.ctx.beginPath();
            this.ctx.arc(x, y, size, 0, Math.PI * 2);
            this.ctx.fill();
        }
        
        // Continue animation
        requestAnimationFrame(() => this.drawMenuBackground());
    }

    startNewGame() {
        this.gameState = 'playing';
        this.moves = 0;
        this.score = 0;
        this.lives = 3;
        this.level = 1;
        this.startTime = Date.now();
        
        // Hide menu, show game UI
        document.getElementById('mazeMainMenu').style.display = 'none';
        document.getElementById('gameHUD').style.display = 'block';
        document.getElementById('mobileControls').style.display = 'block';
        
        this.generateMaze();
        this.initializePlayer();
        this.spawnEnemies();
        this.spawnPowerUps();
        this.startGameLoop();
        
        this.updateHUD();
    }

    generateMaze() {
        const { rows, cols, wallDensity } = this.settings[this.difficulty];
        this.rows = rows;
        this.cols = cols;
        this.cellSize = Math.floor(this.canvas.width / cols);
        
        // Initialize maze
        this.maze = [];
        for (let i = 0; i < rows; i++) {
            this.maze[i] = [];
            for (let j = 0; j < cols; j++) {
                if (i === 0 || j === 0 || i === rows - 1 || j === cols - 1) {
                    this.maze[i][j] = 1; // Wall
                } else {
                    this.maze[i][j] = Math.random() < wallDensity ? 1 : 0;
                }
            }
        }
        
        // Ensure start and end are clear
        this.maze[1][1] = 0; // Start
        this.maze[rows - 2][cols - 2] = 2; // Exit
        
        // Create guaranteed path using simple pathfinding
        this.ensurePath();
        
        // Add special cells based on game mode
        this.addSpecialCells();
    }

    ensurePath() {
        // Simple path creation from start to end
        let currentRow = 1, currentCol = 1;
        const targetRow = this.rows - 2, targetCol = this.cols - 2;
        
        while (currentRow !== targetRow || currentCol !== targetCol) {
            this.maze[currentRow][currentCol] = 0;
            
            if (currentRow < targetRow && Math.random() > 0.5) {
                currentRow++;
            } else if (currentCol < targetCol) {
                currentCol++;
            } else if (currentRow < targetRow) {
                currentRow++;
            }
        }
    }

    addSpecialCells() {
        // Add treasure chests, traps, etc.
        const specialCells = Math.floor(this.rows * this.cols * 0.05);
        
        for (let i = 0; i < specialCells; i++) {
            let row, col;
            do {
                row = Math.floor(Math.random() * (this.rows - 2)) + 1;
                col = Math.floor(Math.random() * (this.cols - 2)) + 1;
            } while (this.maze[row][col] !== 0);
            
            // 3: treasure, 4: trap, 5: teleporter
            this.maze[row][col] = Math.random() < 0.6 ? 3 : (Math.random() < 0.8 ? 4 : 5);
        }
    }

    initializePlayer() {
        this.player = {
            row: 1,
            col: 1,
            x: this.cellSize * 1 + this.cellSize / 2,
            y: this.cellSize * 1 + this.cellSize / 2,
            radius: this.cellSize / 4,
            color: '#ff4444',
            trail: [],
            invulnerable: false,
            invulnerabilityTime: 0
        };
        
        // Reset abilities
        this.playerAbilities = {
            speed: 1,
            wallBreaker: false,
            invisible: false,
            timeFreeze: false,
            shield: false
        };
    }

    spawnEnemies() {
        this.enemies = [];
        
        if (this.gameMode === 'survival' || this.gameMode === 'timed') {
            const enemyCount = Math.floor(this.level * 1.5);
            
            for (let i = 0; i < enemyCount; i++) {
                let row, col;
                do {
                    row = Math.floor(Math.random() * (this.rows - 2)) + 1;
                    col = Math.floor(Math.random() * (this.cols - 2)) + 1;
                } while (this.maze[row][col] !== 0 || (row === 1 && col === 1));
                
                this.enemies.push({
                    row: row,
                    col: col,
                    x: col * this.cellSize + this.cellSize / 2,
                    y: row * this.cellSize + this.cellSize / 2,
                    radius: this.cellSize / 5,
                    color: '#8844ff',
                    direction: Math.floor(Math.random() * 4),
                    moveTimer: 0,
                    type: Math.random() < 0.7 ? 'patrol' : 'hunter'
                });
            }
        }
    }

    spawnPowerUps() {
        this.powerUps = [];
        const powerUpCount = Math.floor(this.rows * this.cols * 0.02);
        
        const powerUpTypes = ['speed', 'wallBreaker', 'invisible', 'timeFreeze', 'shield', 'extraLife'];
        
        for (let i = 0; i < powerUpCount; i++) {
            let row, col;
            do {
                row = Math.floor(Math.random() * (this.rows - 2)) + 1;
                col = Math.floor(Math.random() * (this.cols - 2)) + 1;
            } while (this.maze[row][col] !== 0);
            
            this.powerUps.push({
                row: row,
                col: col,
                x: col * this.cellSize + this.cellSize / 2,
                y: row * this.cellSize + this.cellSize / 2,
                type: powerUpTypes[Math.floor(Math.random() * powerUpTypes.length)],
                collected: false,
                pulseTime: 0
            });
        }
    }

    startGameLoop() {
        this.lastTime = performance.now();
        this.gameLoop();
    }

    gameLoop(currentTime = performance.now()) {
        if (this.gameState !== 'playing') return;
        
        const deltaTime = currentTime - this.lastTime;
        this.lastTime = currentTime;
        
        this.update(deltaTime);
        this.render();
        
        this.animationId = requestAnimationFrame((time) => this.gameLoop(time));
    }

    update(deltaTime) {
        // Update game time
        this.gameTime = (Date.now() - this.startTime) / 1000;
        
        // Update enemies
        this.updateEnemies(deltaTime);
        
        // Update power-ups
        this.updatePowerUps(deltaTime);
        
        // Update particles
        this.updateParticles(deltaTime);
        
        // Update player invulnerability
        if (this.player.invulnerable) {
            this.player.invulnerabilityTime -= deltaTime;
            if (this.player.invulnerabilityTime <= 0) {
                this.player.invulnerable = false;
            }
        }
        
        // Check win/lose conditions
        this.checkGameConditions();
        
        // Update HUD
        this.updateHUD();
    }

    updateEnemies(deltaTime) {
        this.enemies.forEach(enemy => {
            enemy.moveTimer += deltaTime;
            
            if (enemy.moveTimer > 1000) { // Move every second
                enemy.moveTimer = 0;
                
                if (enemy.type === 'hunter') {
                    this.moveEnemyTowardsPlayer(enemy);
                } else {
                    this.moveEnemyPatrol(enemy);
                }
            }
            
            // Check collision with player
            if (!this.player.invulnerable && !this.playerAbilities.invisible) {
                const distance = Math.sqrt(
                    Math.pow(enemy.x - this.player.x, 2) + 
                    Math.pow(enemy.y - this.player.y, 2)
                );
                
                if (distance < enemy.radius + this.player.radius) {
                    this.playerHit();
                }
            }
        });
    }

    moveEnemyTowardsPlayer(enemy) {
        const directions = [
            { row: -1, col: 0 }, // Up
            { row: 1, col: 0 },  // Down
            { row: 0, col: -1 }, // Left
            { row: 0, col: 1 }   // Right
        ];
        
        let bestDirection = null;
        let bestDistance = Infinity;
        
        directions.forEach(dir => {
            const newRow = enemy.row + dir.row;
            const newCol = enemy.col + dir.col;
            
            if (this.isValidMove(newRow, newCol)) {
                const distance = Math.abs(newRow - this.player.row) + Math.abs(newCol - this.player.col);
                if (distance < bestDistance) {
                    bestDistance = distance;
                    bestDirection = dir;
                }
            }
        });
        
        if (bestDirection) {
            enemy.row += bestDirection.row;
            enemy.col += bestDirection.col;
            enemy.x = enemy.col * this.cellSize + this.cellSize / 2;
            enemy.y = enemy.row * this.cellSize + this.cellSize / 2;
        }
    }

    moveEnemyPatrol(enemy) {
        const directions = [
            { row: -1, col: 0 }, // Up
            { row: 1, col: 0 },  // Down
            { row: 0, col: -1 }, // Left
            { row: 0, col: 1 }   // Right
        ];
        
        // Try to continue in current direction
        let dir = directions[enemy.direction];
        let newRow = enemy.row + dir.row;
        let newCol = enemy.col + dir.col;
        
        if (!this.isValidMove(newRow, newCol)) {
            // Change direction randomly
            enemy.direction = Math.floor(Math.random() * 4);
            dir = directions[enemy.direction];
            newRow = enemy.row + dir.row;
            newCol = enemy.col + dir.col;
        }
        
        if (this.isValidMove(newRow, newCol)) {
            enemy.row = newRow;
            enemy.col = newCol;
            enemy.x = enemy.col * this.cellSize + this.cellSize / 2;
            enemy.y = enemy.row * this.cellSize + this.cellSize / 2;
        }
    }

    updatePowerUps(deltaTime) {
        this.powerUps.forEach(powerUp => {
            if (!powerUp.collected) {
                powerUp.pulseTime += deltaTime;
                
                // Check collection
                if (powerUp.row === this.player.row && powerUp.col === this.player.col) {
                    this.collectPowerUp(powerUp);
                }
            }
        });
    }

    updateParticles(deltaTime) {
        this.particles = this.particles.filter(particle => {
            particle.life -= deltaTime;
            particle.x += particle.vx * deltaTime / 1000;
            particle.y += particle.vy * deltaTime / 1000;
            particle.alpha = particle.life / particle.maxLife;
            return particle.life > 0;
        });
    }

    collectPowerUp(powerUp) {
        powerUp.collected = true;
        this.sounds.powerUp();
        this.score += 50;
        
        // Create particles
        this.createParticles(powerUp.x, powerUp.y, '#ffff00', 10);
        
        // Apply power-up effect
        switch (powerUp.type) {
            case 'speed':
                this.playerAbilities.speed = 2;
                setTimeout(() => this.playerAbilities.speed = 1, 10000);
                break;
            case 'wallBreaker':
                this.playerAbilities.wallBreaker = true;
                setTimeout(() => this.playerAbilities.wallBreaker = false, 15000);
                break;
            case 'invisible':
                this.playerAbilities.invisible = true;
                setTimeout(() => this.playerAbilities.invisible = false, 8000);
                break;
            case 'timeFreeze':
                this.playerAbilities.timeFreeze = true;
                setTimeout(() => this.playerAbilities.timeFreeze = false, 5000);
                break;
            case 'shield':
                this.playerAbilities.shield = true;
                setTimeout(() => this.playerAbilities.shield = false, 12000);
                break;
            case 'extraLife':
                this.lives++;
                break;
        }
    }

    createParticles(x, y, color, count) {
        for (let i = 0; i < count; i++) {
            this.particles.push({
                x: x,
                y: y,
                vx: (Math.random() - 0.5) * 200,
                vy: (Math.random() - 0.5) * 200,
                color: color,
                life: 1000,
                maxLife: 1000,
                alpha: 1,
                size: Math.random() * 4 + 2
            });
        }
    }

    playerHit() {
        if (this.playerAbilities.shield) {
            this.playerAbilities.shield = false;
            return;
        }
        
        this.lives--;
        this.player.invulnerable = true;
        this.player.invulnerabilityTime = 2000;
        this.sounds.defeat();
        
        // Create red particles
        this.createParticles(this.player.x, this.player.y, '#ff0000', 15);
        
        if (this.lives <= 0) {
            this.gameOver();
        }
    }

    checkGameConditions() {
        // Check if player reached exit
        if (this.maze[this.player.row][this.player.col] === 2) {
            this.levelComplete();
        }
        
        // Check time limit for timed mode
        if (this.gameMode === 'timed') {
            const timeLimit = this.settings[this.difficulty].timeLimit;
            if (this.gameTime > timeLimit) {
                this.gameOver();
            }
        }
    }

    levelComplete() {
        this.sounds.victory();
        this.score += 1000 + Math.floor((this.settings[this.difficulty].timeLimit - this.gameTime) * 10);
        this.level++;
        
        // Create victory particles
        this.createParticles(this.player.x, this.player.y, '#00ff00', 20);
        
        // Generate new level
        setTimeout(() => {
            this.generateMaze();
            this.initializePlayer();
            this.spawnEnemies();
            this.spawnPowerUps();
        }, 2000);
    }

    gameOver() {
        this.gameState = 'gameOver';
        this.recordScore();
        
        // Show game over screen
        setTimeout(() => {
            alert(`Game Over! Final Score: ${this.score}\nLevel Reached: ${this.level}`);
            this.showMainMenu();
        }, 1000);
    }

    handleKeyPress(e) {
        if (this.gameState !== 'playing') return;
        
        switch (e.key) {
            case 'ArrowUp':
            case 'w':
            case 'W':
                this.movePlayer(0, -1);
                break;
            case 'ArrowDown':
            case 's':
            case 'S':
                this.movePlayer(0, 1);
                break;
            case 'ArrowLeft':
            case 'a':
            case 'A':
                this.movePlayer(-1, 0);
                break;
            case 'ArrowRight':
            case 'd':
            case 'D':
                this.movePlayer(1, 0);
                break;
            case ' ':
                this.usePlayerAction();
                break;
            case 'p':
            case 'P':
                this.togglePause();
                break;
        }
        
        e.preventDefault();
    }

    movePlayer(deltaCol, deltaRow) {
        if (this.gameState !== 'playing') return;
        
        const newRow = this.player.row + deltaRow;
        const newCol = this.player.col + deltaCol;
        
        // Check bounds
        if (newRow < 0 || newRow >= this.rows || newCol < 0 || newCol >= this.cols) {
            return;
        }
        
        // Check wall collision
        if (this.maze[newRow][newCol] === 1 && !this.playerAbilities.wallBreaker) {
            return;
        }
        
        // Move player
        this.player.row = newRow;
        this.player.col = newCol;
        this.player.x = newCol * this.cellSize + this.cellSize / 2;
        this.player.y = newRow * this.cellSize + this.cellSize / 2;
        
        // Add to trail
        this.player.trail.push({ x: this.player.x, y: this.player.y });
        if (this.player.trail.length > 20) {
            this.player.trail.shift();
        }
        
        this.moves++;
        this.score += 1;
        this.sounds.move();
        
        // Handle special cells
        this.handleSpecialCell(newRow, newCol);
    }

    handleSpecialCell(row, col) {
        const cellType = this.maze[row][col];
        
        switch (cellType) {
            case 3: // Treasure
                this.score += 100;
                this.sounds.collect();
                this.createParticles(this.player.x, this.player.y, '#ffd700', 8);
                this.maze[row][col] = 0;
                break;
            case 4: // Trap
                if (!this.playerAbilities.shield) {
                    this.playerHit();
                }
                break;
            case 5: // Teleporter
                this.teleportPlayer();
                break;
        }
    }

    teleportPlayer() {
        let newRow, newCol;
        do {
            newRow = Math.floor(Math.random() * (this.rows - 2)) + 1;
            newCol = Math.floor(Math.random() * (this.cols - 2)) + 1;
        } while (this.maze[newRow][newCol] === 1);
        
        this.player.row = newRow;
        this.player.col = newCol;
        this.player.x = newCol * this.cellSize + this.cellSize / 2;
        this.player.y = newRow * this.cellSize + this.cellSize / 2;
        
        this.createParticles(this.player.x, this.player.y, '#ff00ff', 12);
    }

    usePlayerAction() {
        // Special action based on current abilities
        if (this.playerAbilities.wallBreaker) {
            // Break nearby walls
            for (let dr = -1; dr <= 1; dr++) {
                for (let dc = -1; dc <= 1; dc++) {
                    const r = this.player.row + dr;
                    const c = this.player.col + dc;
                    if (r >= 0 && r < this.rows && c >= 0 && c < this.cols && this.maze[r][c] === 1) {
                        this.maze[r][c] = 0;
                        this.createParticles(c * this.cellSize + this.cellSize/2, r * this.cellSize + this.cellSize/2, '#888888', 5);
                    }
                }
            }
        }
    }

    isValidMove(row, col) {
        return row >= 0 && row < this.rows && col >= 0 && col < this.cols && this.maze[row][col] !== 1;
    }

    togglePause() {
        if (this.gameState === 'playing') {
            this.gameState = 'paused';
            cancelAnimationFrame(this.animationId);
        } else if (this.gameState === 'paused') {
            this.gameState = 'playing';
            this.startGameLoop();
        }
    }

    render() {
        // Clear canvas
        this.ctx.fillStyle = '#0a0a0a';
        this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
        
        // Draw maze
        this.drawMaze();
        
        // Draw power-ups
        this.drawPowerUps();
        
        // Draw enemies
        this.drawEnemies();
        
        // Draw player trail
        this.drawPlayerTrail();
        
        // Draw player
        this.drawPlayer();
        
        // Draw particles
        this.drawParticles();
        
        // Draw effects
        this.drawEffects();
    }

    drawMaze() {
        for (let row = 0; row < this.rows; row++) {
            for (let col = 0; col < this.cols; col++) {
                const x = col * this.cellSize;
                const y = row * this.cellSize;
                
                switch (this.maze[row][col]) {
                    case 1: // Wall
                        this.ctx.fillStyle = '#333333';
                        this.ctx.fillRect(x, y, this.cellSize, this.cellSize);
                        // Add wall texture
                        this.ctx.fillStyle = '#444444';
                        this.ctx.fillRect(x + 2, y + 2, this.cellSize - 4, this.cellSize - 4);
                        break;
                    case 2: // Exit
                        this.ctx.fillStyle = '#00ff00';
                        this.ctx.fillRect(x, y, this.cellSize, this.cellSize);
                        // Draw exit symbol
                        this.ctx.fillStyle = '#ffffff';
                        this.ctx.font = `${this.cellSize/2}px Arial`;
                        this.ctx.textAlign = 'center';
                        this.ctx.fillText('🚪', x + this.cellSize/2, y + this.cellSize/2 + this.cellSize/6);
                        break;
                    case 3: // Treasure
                        this.ctx.fillStyle = '#ffd700';
                        this.ctx.fillRect(x, y, this.cellSize, this.cellSize);
                        this.ctx.fillStyle = '#ffffff';
                        this.ctx.font = `${this.cellSize/2}px Arial`;
                        this.ctx.textAlign = 'center';
                        this.ctx.fillText('💰', x + this.cellSize/2, y + this.cellSize/2 + this.cellSize/6);
                        break;
                    case 4: // Trap
                        this.ctx.fillStyle = '#ff4444';
                        this.ctx.fillRect(x, y, this.cellSize, this.cellSize);
                        this.ctx.fillStyle = '#ffffff';
                        this.ctx.font = `${this.cellSize/2}px Arial`;
                        this.ctx.textAlign = 'center';
                        this.ctx.fillText('⚠️', x + this.cellSize/2, y + this.cellSize/2 + this.cellSize/6);
                        break;
                    case 5: // Teleporter
                        this.ctx.fillStyle = '#ff00ff';
                        this.ctx.fillRect(x, y, this.cellSize, this.cellSize);
                        this.ctx.fillStyle = '#ffffff';
                        this.ctx.font = `${this.cellSize/2}px Arial`;
                        this.ctx.textAlign = 'center';
                        this.ctx.fillText('🌀', x + this.cellSize/2, y + this.cellSize/2 + this.cellSize/6);
                        break;
                    default: // Empty space
                        this.ctx.fillStyle = '#111111';
                        this.ctx.fillRect(x, y, this.cellSize, this.cellSize);
                        break;
                }
                
                // Draw grid lines
                this.ctx.strokeStyle = '#222222';
                this.ctx.lineWidth = 1;
                this.ctx.strokeRect(x, y, this.cellSize, this.cellSize);
            }
        }
    }

    drawPowerUps() {
        this.powerUps.forEach(powerUp => {
            if (!powerUp.collected) {
                const pulse = Math.sin(powerUp.pulseTime / 200) * 0.3 + 0.7;
                const size = this.cellSize / 3 * pulse;
                
                // Power-up colors
                const colors = {
                    speed: '#00ffff',
                    wallBreaker: '#ff8800',
                    invisible: '#8888ff',
                    timeFreeze: '#ffff00',
                    shield: '#00ff88',
                    extraLife: '#ff0088'
                };
                
                this.ctx.fillStyle = colors[powerUp.type] || '#ffffff';
                this.ctx.beginPath();
                this.ctx.arc(powerUp.x, powerUp.y, size, 0, Math.PI * 2);
                this.ctx.fill();
                
                // Draw power-up symbol
                this.ctx.fillStyle = '#000000';
                this.ctx.font = `${size}px Arial`;
                this.ctx.textAlign = 'center';
                const symbols = {
                    speed: '⚡',
                    wallBreaker: '🔨',
                    invisible: '👻',
                    timeFreeze: '❄️',
                    shield: '🛡️',
                    extraLife: '❤️'
                };
                this.ctx.fillText(symbols[powerUp.type] || '?', powerUp.x, powerUp.y + size/3);
            }
        });
    }

    drawEnemies() {
        this.enemies.forEach(enemy => {
            // Enemy glow effect
            const gradient = this.ctx.createRadialGradient(
                enemy.x, enemy.y, 0,
                enemy.x, enemy.y, enemy.radius * 2
            );
            gradient.addColorStop(0, enemy.color);
            gradient.addColorStop(1, 'transparent');
            
            this.ctx.fillStyle = gradient;
            this.ctx.beginPath();
            this.ctx.arc(enemy.x, enemy.y, enemy.radius * 2, 0, Math.PI * 2);
            this.ctx.fill();
            
            // Enemy body
            this.ctx.fillStyle = enemy.color;
            this.ctx.beginPath();
            this.ctx.arc(enemy.x, enemy.y, enemy.radius, 0, Math.PI * 2);
            this.ctx.fill();
            
            // Enemy eyes
            this.ctx.fillStyle = '#ffffff';
            this.ctx.beginPath();
            this.ctx.arc(enemy.x - enemy.radius/3, enemy.y - enemy.radius/3, enemy.radius/4, 0, Math.PI * 2);
            this.ctx.arc(enemy.x + enemy.radius/3, enemy.y - enemy.radius/3, enemy.radius/4, 0, Math.PI * 2);
            this.ctx.fill();
        });
    }

    drawPlayerTrail() {
        this.player.trail.forEach((point, index) => {
            const alpha = index / this.player.trail.length * 0.5;
            this.ctx.fillStyle = `rgba(255, 68, 68, ${alpha})`;
            this.ctx.beginPath();
            this.ctx.arc(point.x, point.y, this.player.radius * 0.3, 0, Math.PI * 2);
            this.ctx.fill();
        });
    }

    drawPlayer() {
        // Player glow effect
        if (this.player.invulnerable) {
            const pulse = Math.sin(Date.now() / 100) * 0.5 + 0.5;
            this.ctx.fillStyle = `rgba(255, 255, 255, ${pulse * 0.5})`;
            this.ctx.beginPath();
            this.ctx.arc(this.player.x, this.player.y, this.player.radius * 2, 0, Math.PI * 2);
            this.ctx.fill();
        }
        
        // Player body
        let playerColor = this.player.color;
        if (this.playerAbilities.invisible) {
            playerColor = 'rgba(255, 68, 68, 0.3)';
        } else if (this.playerAbilities.shield) {
            playerColor = '#00ffff';
        }
        
        this.ctx.fillStyle = playerColor;
        this.ctx.beginPath();
        this.ctx.arc(this.player.x, this.player.y, this.player.radius, 0, Math.PI * 2);
        this.ctx.fill();
        
        // Player face
        this.ctx.fillStyle = '#ffffff';
        this.ctx.beginPath();
        this.ctx.arc(this.player.x - this.player.radius/3, this.player.y - this.player.radius/3, this.player.radius/5, 0, Math.PI * 2);
        this.ctx.arc(this.player.x + this.player.radius/3, this.player.y - this.player.radius/3, this.player.radius/5, 0, Math.PI * 2);
        this.ctx.fill();
        
        // Player mouth
        this.ctx.strokeStyle = '#ffffff';
        this.ctx.lineWidth = 2;
        this.ctx.beginPath();
        this.ctx.arc(this.player.x, this.player.y + this.player.radius/4, this.player.radius/3, 0, Math.PI);
        this.ctx.stroke();
    }

    drawParticles() {
        this.particles.forEach(particle => {
            this.ctx.fillStyle = particle.color.replace(')', `, ${particle.alpha})`).replace('rgb', 'rgba');
            this.ctx.beginPath();
            this.ctx.arc(particle.x, particle.y, particle.size, 0, Math.PI * 2);
            this.ctx.fill();
        });
    }

    drawEffects() {
        // Time freeze effect
        if (this.playerAbilities.timeFreeze) {
            this.ctx.fillStyle = 'rgba(0, 255, 255, 0.1)';
            this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
        }
        
        // Speed effect
        if (this.playerAbilities.speed > 1) {
            const gradient = this.ctx.createRadialGradient(
                this.player.x, this.player.y, 0,
                this.player.x, this.player.y, this.player.radius * 3
            );
            gradient.addColorStop(0, 'transparent');
            gradient.addColorStop(1, 'rgba(255, 255, 0, 0.3)');
            
            this.ctx.fillStyle = gradient;
            this.ctx.beginPath();
            this.ctx.arc(this.player.x, this.player.y, this.player.radius * 3, 0, Math.PI * 2);
            this.ctx.fill();
        }
    }

    updateHUD() {
        document.getElementById('scoreDisplay').textContent = this.score;
        document.getElementById('levelDisplay').textContent = this.level;
        
        // Update time display
        const minutes = Math.floor(this.gameTime / 60);
        const seconds = Math.floor(this.gameTime % 60);
        document.getElementById('timeDisplay').textContent = `${minutes}:${seconds.toString().padStart(2, '0')}`;
        
        // Update lives display
        const heartsDisplay = '❤️'.repeat(this.lives) + '🖤'.repeat(Math.max(0, 3 - this.lives));
        document.getElementById('livesDisplay').textContent = heartsDisplay;
        
        // Update active power-ups
        const activePowerUps = document.getElementById('activePowerUps');
        activePowerUps.innerHTML = '';
        
        Object.entries(this.playerAbilities).forEach(([ability, active]) => {
            if (active && ability !== 'speed') {
                const powerUpIcon = document.createElement('span');
                powerUpIcon.className = 'power-up-icon';
                powerUpIcon.textContent = {
                    wallBreaker: '🔨',
                    invisible: '👻',
                    timeFreeze: '❄️',
                    shield: '🛡️'
                }[ability] || '⭐';
                activePowerUps.appendChild(powerUpIcon);
            }
        });
    }

    recordScore() {
        const scores = JSON.parse(localStorage.getItem('mazeHighScores')) || [];
        scores.push({
            score: this.score,
            level: this.level,
            time: this.gameTime,
            mode: this.gameMode,
            difficulty: this.difficulty,
            date: new Date().toISOString()
        });
        
        scores.sort((a, b) => b.score - a.score);
        scores.splice(10); // Keep only top 10
        
        localStorage.setItem('mazeHighScores', JSON.stringify(scores));
    }

    loadAchievements() {
        // Load achievements from localStorage
        this.achievements = JSON.parse(localStorage.getItem('mazeAchievements')) || [];
    }
}

// Initialize the enhanced maze game when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    // Add CSS styles
    const style = document.createElement('style');
    style.textContent = `
        .maze-menu {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            margin: 20px 0;
            text-align: center;
        }
        
        .menu-section {
            margin: 15px 0;
        }
        
        .menu-section h3 {
            margin: 10px 0 5px 0;
        }
        
        .menu-section select {
            padding: 8px 12px;
            border-radius: 5px;
            border: none;
            font-size: 14px;
        }
        
        .menu-buttons {
            display: flex;
            gap: 10px;
            justify-content: center;
            flex-wrap: wrap;
            margin-top: 20px;
        }
        
        .btn-primary {
            background: #ff6b6b;
            color: white;
            border: none;
            padding: 12px 20px;
            border-radius: 25px;
            font-size: 16px;
            font-weight: bold;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        
        .btn-primary:hover {
            background: #ff5252;
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        }
        
        .btn-secondary {
            background: #4ecdc4;
            color: white;
            border: none;
            padding: 10px 16px;
            border-radius: 20px;
            font-size: 14px;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        
        .btn-secondary:hover {
            background: #45b7aa;
            transform: translateY(-1px);
        }
        
        .game-hud {
            background: rgba(0, 0, 0, 0.8);
            color: white;
            padding: 15px;
            border-radius: 10px;
            margin: 10px 0;
        }
        
        .hud-top {
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 15px;
        }
        
        .hud-item {
            display: flex;
            align-items: center;
            gap: 5px;
        }
        
        .hud-label {
            font-weight: bold;
            color: #ccc;
        }
        
        .hud-bottom {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-top: 10px;
        }
        
        .power-ups {
            display: flex;
            gap: 5px;
        }
        
        .power-up-icon {
            font-size: 20px;
            animation: pulse 1s infinite;
        }
        
        @keyframes pulse {
            0%, 100% { transform: scale(1); }
            50% { transform: scale(1.1); }
        }
        
        .game-controls {
            display: flex;
            gap: 10px;
        }
        
        .btn-small {
            background: #666;
            color: white;
            border: none;
            padding: 8px 12px;
            border-radius: 15px;
            font-size: 12px;
            cursor: pointer;
            transition: background 0.3s ease;
        }
        
        .btn-small:hover {
            background: #777;
        }
        
        .mobile-controls {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin: 20px 0;
            padding: 15px;
            background: rgba(0, 0, 0, 0.1);
            border-radius: 10px;
        }
        
        .control-pad {
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 5px;
        }
        
        .control-row {
            display: flex;
            gap: 5px;
        }
        
        .control-btn {
            width: 50px;
            height: 50px;
            background: #4ecdc4;
            color: white;
            border: none;
            border-radius: 10px;
            font-size: 20px;
            font-weight: bold;
            cursor: pointer;
            transition: all 0.2s ease;
            user-select: none;
        }
        
        .control-btn:hover, .control-btn:active {
            background: #45b7aa;
            transform: scale(0.95);
        }
        
        .action-btn {
            padding: 12px 20px;
            background: #ff6b6b;
            color: white;
            border: none;
            border-radius: 25px;
            font-size: 14px;
            font-weight: bold;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        
        .action-btn:hover {
            background: #ff5252;
            transform: translateY(-2px);
        }
        
        @media (max-width: 768px) {
            .menu-buttons {
                flex-direction: column;
                align-items: center;
            }
            
            .btn-primary, .btn-secondary {
                width: 200px;
            }
            
            .hud-top {
                flex-direction: column;
                gap: 10px;
            }
            
            .mobile-controls {
                flex-direction: column;
                gap: 15px;
            }
        }
    `;
    document.head.appendChild(style);
    
    // Initialize the enhanced maze game
    window.enhancedMazeGame = new EnhancedMazeGame();
});