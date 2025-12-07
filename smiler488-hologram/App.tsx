import React, { useState } from 'react';
import HologramParticles from './components/HologramParticles';
import { HandData } from './types';
import { FaHandPaper, FaHandRock, FaFingerprint } from 'react-icons/fa';

const App: React.FC = () => {
  const [status, setStatus] = useState<string>("INITIALIZING...");
  const [handData, setHandData] = useState<HandData | null>(null);

  return (
    <div className="relative w-full h-screen font-sci text-holo-300 overflow-hidden cursor-crosshair">
      
      {/* Background Hologram Layer (Includes Camera Feed) */}
      <HologramParticles 
        text="SMILER488" 
        onStatusChange={setStatus} 
        onHandDetect={setHandData}
      />

      {/* Foreground UI Layer (HUD) */}
      <div className="absolute inset-0 z-30 pointer-events-none flex flex-col justify-between p-6 md:p-12">
        
        {/* Top Header */}
        <div className="flex justify-between items-start animate-fade-in-down">
          <div>
            <div className="flex items-center gap-3 mb-2">
               <FaFingerprint className="text-2xl animate-pulse text-holo-400" />
               <h1 className="text-2xl md:text-4xl font-bold tracking-[0.2em] text-white drop-shadow-[0_0_10px_rgba(45,212,191,0.8)]">
                 IDENTITY: SMILER488
               </h1>
            </div>
            <div className="h-0.5 w-64 bg-gradient-to-r from-holo-500 via-holo-300 to-transparent"></div>
            <div className="mt-1 text-xs font-tech text-holo-500 tracking-widest uppercase">
              Secure Holographic Terminal v4.8.8
            </div>
          </div>

          {/* System Status Top Right */}
          <div className="text-right hidden md:block">
            <div className="flex flex-col gap-1 text-[10px] font-tech text-holo-600 bg-black/60 p-2 rounded backdrop-blur-sm border border-holo-900">
               <span className="flex justify-between w-24"><span>MEM:</span> <span className="text-holo-400">64TB</span></span>
               <span className="flex justify-between w-24"><span>CAM:</span> <span className="text-holo-400">ACTIVE</span></span>
               <span className="flex justify-between w-24"><span>NET:</span> <span className="text-holo-400">SECURE</span></span>
            </div>
          </div>
        </div>

        {/* Center Prompt - only visible when no hand */}
        {!handData && (
           <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 text-center pointer-events-none">
              <div className="animate-pulse flex flex-col items-center gap-4 opacity-70">
                 <div className="w-16 h-16 border border-holo-500 rounded-full flex items-center justify-center relative">
                    <span className="absolute w-full h-full rounded-full border border-holo-500 animate-ping opacity-20"></span>
                    <FaHandPaper className="text-2xl text-holo-300" />
                 </div>
                 <div className="bg-black/40 backdrop-blur-sm p-4 border-l-2 border-r-2 border-holo-500/50">
                    <p className="text-holo-100 font-bold tracking-widest text-lg">SYSTEM LOCKED</p>
                    <p className="text-holo-500 text-xs mt-1 font-tech uppercase">Raise hand to interface with particles</p>
                 </div>
              </div>
           </div>
        )}

        {/* Bottom HUD */}
        <div className="flex justify-between items-end">
          
          {/* Gesture Feedback Panel */}
          <div className="flex flex-col gap-2">
            <div className="text-[10px] text-holo-600 font-mono mb-1 bg-black/50 inline-block px-2">GESTURE RECOGNITION</div>
            
            <div className={`transition-all duration-500 transform ${handData ? 'translate-y-0 opacity-100' : 'translate-y-10 opacity-0'}`}>
               <div className="flex items-center gap-4 bg-black/80 border border-holo-500/30 p-4 rounded-tr-xl backdrop-blur-md shadow-[0_0_20px_rgba(45,212,191,0.1)]">
                  {handData?.isClosed ? (
                    <div className="flex items-center gap-3">
                      <div className="p-2 bg-red-500/20 rounded border border-red-500/50">
                        <FaHandRock className="text-2xl text-red-400" />
                      </div>
                      <div>
                        <div className="text-red-400 font-bold tracking-widest text-sm">GRAVITY WELL</div>
                        <div className="text-[10px] text-red-500/80 uppercase">Absorbing Matter</div>
                      </div>
                    </div>
                  ) : (
                    <div className="flex items-center gap-3">
                      <div className="p-2 bg-holo-500/20 rounded border border-holo-500/50">
                        <FaHandPaper className="text-2xl text-holo-300" />
                      </div>
                      <div>
                        <div className="text-holo-300 font-bold tracking-widest text-sm">REPULSOR FIELD</div>
                        <div className="text-[10px] text-holo-500/80 uppercase">Scattering Matter</div>
                      </div>
                    </div>
                  )}
               </div>
            </div>
          </div>

          {/* Coordinates & Status */}
          <div className="flex flex-col items-end gap-1">
             <div className="flex items-center gap-2 text-xs font-tech text-holo-400 uppercase tracking-widest bg-black/70 px-3 py-1 rounded border border-holo-900/50">
               <div className={`w-2 h-2 rounded-full ${handData ? 'bg-holo-400 animate-pulse' : 'bg-yellow-500'}`}></div>
               {status}
             </div>
             <div className="text-[10px] text-holo-800 font-mono bg-black/40 px-2 rounded">
                X: {handData?.x.toFixed(0) || '---'} | Y: {handData?.y.toFixed(0) || '---'}
             </div>
          </div>

        </div>
      </div>
    </div>
  );
};

export default App;
