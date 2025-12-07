import React, { useEffect, useRef } from 'react';
import { initializeHandLandmarker } from '../services/visionService';
import { HandData } from '../types';

interface HologramParticlesProps {
  text: string;
  onStatusChange: (status: string) => void;
  onHandDetect: (data: HandData | null) => void;
}

const HologramParticles: React.FC<HologramParticlesProps> = ({ text, onStatusChange, onHandDetect }) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  
  // Advanced Physics Constants
  const FRICTION = 0.94; 
  const EASE = 0.05; 
  const RADIUS_ATTRACT = 300;
  const RADIUS_REPEL = 200;
  
  useEffect(() => {
    let animationFrameId: number;
    let particles: Particle[] = [];
    let mouse = { x: -10000, y: -10000 };
    let handData: HandData | null = null;
    let handLandmarker: any = null;

    class Particle {
      x: number;
      y: number;
      z: number; // Depth for 3D effect
      baseX: number;
      baseY: number;
      vx: number;
      vy: number;
      color: string;
      size: number;
      density: number;

      constructor(x: number, y: number, color: string) {
        this.baseX = x;
        this.baseY = y;
        this.x = Math.random() * window.innerWidth;
        this.y = Math.random() * window.innerHeight;
        // Z-depth simulation (0 to 1)
        this.z = Math.random(); 
        this.vx = 0;
        this.vy = 0;
        this.color = color;
        // Particles closer (higher z) are larger and move faster (parallax)
        this.size = 1.5 + this.z; 
        this.density = (Math.random() * 30) + 1;
      }

      draw(ctx: CanvasRenderingContext2D) {
        ctx.fillStyle = this.color;
        // Alpha based on depth
        ctx.globalAlpha = 0.6 + (this.z * 0.4); 
        ctx.beginPath();
        ctx.arc(this.x, this.y, this.size, 0, Math.PI * 2);
        ctx.closePath();
        ctx.fill();
        ctx.globalAlpha = 1;
      }

      update() {
        let interactionX = -10000;
        let interactionY = -10000;
        let isActive = false;
        let isAttracting = false;

        // Determine input source (Hand > Mouse)
        if (handData) {
          interactionX = handData.x;
          interactionY = handData.y;
          isActive = true;
          isAttracting = handData.isClosed;
        } else if (mouse.x > 0) {
          interactionX = mouse.x;
          interactionY = mouse.y;
          isActive = true;
          isAttracting = false; // Mouse always repels
        }

        if (isActive) {
          const dx = interactionX - this.x;
          const dy = interactionY - this.y;
          const distance = Math.sqrt(dx * dx + dy * dy);
          
          // Interaction Radius calculation
          const radius = isAttracting ? RADIUS_ATTRACT : RADIUS_REPEL;
          
          if (distance < radius) {
            const forceDirectionX = dx / distance;
            const forceDirectionY = dy / distance;
            
            // Physics Formula:
            const maxDistance = radius;
            const force = (maxDistance - distance) / maxDistance;
            
            // Direction: 1 = Attract (Pull), -1 = Repel (Push)
            const direction = isAttracting ? 1 : -1;
            const power = isAttracting ? 40 : 15; 
            
            // Parallax: Deep particles (low z) move slower
            const parallax = 0.5 + this.z; 

            const forceX = forceDirectionX * force * this.density * parallax * direction * power;
            const forceY = forceDirectionY * force * this.density * parallax * direction * power;

            this.vx += forceX;
            this.vy += forceY;
          }
        }

        // Return to Home Force (Spring)
        const returnForce = isAttracting ? EASE * 0.1 : EASE;
        
        const dxBase = this.baseX - this.x;
        const dyBase = this.baseY - this.y;
        
        this.vx += dxBase * returnForce;
        this.vy += dyBase * returnForce;

        // Friction
        this.vx *= FRICTION;
        this.vy *= FRICTION;

        // Apply velocity
        this.x += this.vx;
        this.y += this.vy;
      }
    }

    const init = async () => {
      onStatusChange("INITIALIZING OPTICS...");
      if (!canvasRef.current || !containerRef.current) return;

      const canvas = canvasRef.current;
      const ctx = canvas.getContext('2d');
      if (!ctx) return;

      const resize = () => {
        canvas.width = containerRef.current?.offsetWidth || window.innerWidth;
        canvas.height = containerRef.current?.offsetHeight || window.innerHeight;
        // Debounce particle recreation
        setTimeout(createParticles, 100);
      };

      const createParticles = () => {
        particles = [];
        ctx.clearRect(0,0, canvas.width, canvas.height);
        
        // Dynamic font size
        let fontSize = Math.min(canvas.width / 5, 250); // Big text
        ctx.font = `900 ${fontSize}px "Orbitron"`;
        ctx.fillStyle = 'white';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(text, canvas.width / 2, canvas.height / 2);

        const textCoordinates = ctx.getImageData(0, 0, canvas.width, canvas.height);
        const gap = 6; // Optimization: Slightly larger gap for performance with video
        
        for (let y = 0; y < textCoordinates.height; y += gap) {
          for (let x = 0; x < textCoordinates.width; x += gap) {
            // Check alpha value of pixel
            if (textCoordinates.data[(y * 4 * textCoordinates.width) + (x * 4) + 3] > 128) {
              const random = Math.random();
              // Palette: Cyan/Teal/White - brighter for AR
              let color = '#2dd4bf'; // Base
              if (random > 0.7) color = '#ffffff'; // White Highlight for visibility
              else if (random < 0.2) color = '#00ffcc'; // Neon Green/Blue
              
              particles.push(new Particle(x, y, color));
            }
          }
        }
        ctx.clearRect(0,0, canvas.width, canvas.height);
      };

      window.addEventListener('resize', resize);
      resize();

      // --- MediaPipe Hand Tracking Setup ---
      try {
        onStatusChange("CONNECTING SENSORS...");
        const stream = await navigator.mediaDevices.getUserMedia({ 
          video: { 
            width: { ideal: 1280 }, 
            height: { ideal: 720 }, 
            frameRate: { ideal: 30 } 
          } 
        });
        
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          videoRef.current.onloadeddata = async () => {
             handLandmarker = await initializeHandLandmarker();
             onStatusChange("VISUAL UPLINK ESTABLISHED");
             videoRef.current?.play();
             predictWebcam();
          };
        }
      } catch (err) {
        console.error("Camera failed", err);
        onStatusChange("CAMERA ERROR - MOUSE MODE ONLY");
      }

      const predictWebcam = () => {
        if (!videoRef.current || !handLandmarker) return;
        
        const now = performance.now();
        const results = handLandmarker.detectForVideo(videoRef.current, now);

        if (results.landmarks && results.landmarks.length > 0) {
           const lm = results.landmarks[0];
           
           // 1. Calculate Hand Center
           const rawX = (lm[0].x + lm[9].x) / 2;
           const rawY = (lm[0].y + lm[9].y) / 2;
           
           // Mirror x-axis logic matches the CSS mirror flip
           const x = (1 - rawX) * canvas.width;
           const y = rawY * canvas.height;

           // 2. Gesture Recognition
           const thumb = lm[4];
           const index = lm[8];
           const middle = lm[12];
           
           const d1 = Math.hypot(thumb.x - index.x, thumb.y - index.y);
           const d2 = Math.hypot(thumb.x - middle.x, thumb.y - middle.y);
           
           const isClosed = (d1 < 0.08 || d2 < 0.08);

           handData = { x, y, isClosed };
           onHandDetect(handData);
           onStatusChange(isClosed ? "TARGET LOCKED: ABSORBING" : "TARGET ACQUIRED: DISPERSING");
        } else {
           handData = null;
           onHandDetect(null);
           onStatusChange("SCANNING SECTOR...");
        }
        
        requestAnimationFrame(predictWebcam);
      };

      // --- Animation Loop ---
      const animate = () => {
        // AR MODE: Clear rect entirely to see video behind
        // Previously we used fillRect with opacity for trails, but that darkens the video.
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        
        particles.forEach(p => {
          p.update();
          p.draw(ctx);
        });

        // Optional: Draw a target reticle on the hand for feedback
        if (handData) {
            ctx.beginPath();
            ctx.strokeStyle = handData.isClosed ? 'rgba(255, 50, 50, 0.8)' : 'rgba(45, 212, 191, 0.8)';
            ctx.lineWidth = 2;
            ctx.arc(handData.x, handData.y, 40, 0, Math.PI * 2);
            ctx.stroke();
            
            // Crosshairs
            ctx.beginPath();
            ctx.moveTo(handData.x - 50, handData.y);
            ctx.lineTo(handData.x + 50, handData.y);
            ctx.moveTo(handData.x, handData.y - 50);
            ctx.lineTo(handData.x, handData.y + 50);
            ctx.stroke();
        }
        
        animationFrameId = requestAnimationFrame(animate);
      };
      animate();

      const handleMouseMove = (e: MouseEvent) => {
        if (!handData) {
          mouse.x = e.clientX;
          mouse.y = e.clientY;
        }
      };
      const handleMouseLeave = () => {
        mouse.x = -10000;
        mouse.y = -10000;
      };
      window.addEventListener('mousemove', handleMouseMove);
      window.addEventListener('mouseout', handleMouseLeave);

      return () => {
        window.removeEventListener('resize', resize);
        window.removeEventListener('mousemove', handleMouseMove);
        window.removeEventListener('mouseout', handleMouseLeave);
        cancelAnimationFrame(animationFrameId);
        if (videoRef.current && videoRef.current.srcObject) {
           const tracks = (videoRef.current.srcObject as MediaStream).getTracks();
           tracks.forEach(t => t.stop());
        }
      };
    };

    init();

  }, [text]);

  return (
    <div ref={containerRef} className="absolute inset-0 z-0 overflow-hidden">
      {/* 
        AR VIDEO BACKGROUND 
        - object-cover: Fills screen
        - -scale-x-100: Mirrors the webcam so left is left, right is right
        - brightness/contrast: Sci-fi look
      */}
      <video 
        ref={videoRef} 
        autoPlay 
        playsInline 
        muted 
        className="absolute inset-0 w-full h-full object-cover transform -scale-x-100 filter brightness-75 contrast-125 saturate-50"
      />
      
      {/* Canvas Layer - Transparent */}
      <canvas 
        ref={canvasRef} 
        className="absolute inset-0 w-full h-full block z-10"
      />
      
      {/* CRT Overlay Effects - Made subtler for AR visibility */}
      <div className="absolute inset-0 pointer-events-none z-20" style={{
        background: 'radial-gradient(circle at center, transparent 40%, rgba(0,0,0,0.6) 100%)'
      }} />
      <div className="absolute inset-0 pointer-events-none opacity-10 z-20" style={{
        backgroundImage: 'linear-gradient(rgba(18, 16, 16, 0) 50%, rgba(0, 0, 0, 0.25) 50%), linear-gradient(90deg, rgba(255, 0, 0, 0.06), rgba(0, 255, 0, 0.02), rgba(0, 0, 255, 0.06))',
        backgroundSize: '100% 2px, 3px 100%'
      }} />
    </div>
  );
};

export default HologramParticles;
