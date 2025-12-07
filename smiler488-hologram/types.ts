// Particle System Types
export interface Point {
  x: number;
  y: number;
}

export interface HandData {
  x: number;
  y: number;
  isClosed: boolean; // True if fist/pinch, False if open palm
}

export enum HologramStatus {
  INITIALIZING = 'INITIALIZING SYSTEM...',
  SCANNING = 'SEARCHING FOR BIOMETRICS...',
  ACTIVE = 'INTERACTIVE MODE ENGAGED',
  ERROR = 'SENSOR MALFUNCTION',
}