import { create } from "zustand";
import type { StrategyLeg, StrategyAnalysis } from "@/lib/types";

interface StrategyState {
  legs: StrategyLeg[];
  analysis: StrategyAnalysis | null;
  timeSliderValue: number; // 0 to 1 (now to expiry)
  isAnalyzing: boolean;

  addLeg: (leg: StrategyLeg) => void;
  removeLeg: (index: number) => void;
  updateLeg: (index: number, leg: StrategyLeg) => void;
  setLegs: (legs: StrategyLeg[]) => void;
  setAnalysis: (analysis: StrategyAnalysis | null) => void;
  setTimeSlider: (value: number) => void;
  setIsAnalyzing: (v: boolean) => void;
  clearStrategy: () => void;
}

export const useStrategyStore = create<StrategyState>((set) => ({
  legs: [],
  analysis: null,
  timeSliderValue: 1, // Default to expiry view
  isAnalyzing: false,

  addLeg: (leg) =>
    set((state) => ({ legs: [...state.legs, leg] })),

  removeLeg: (index) =>
    set((state) => ({
      legs: state.legs.filter((_, i) => i !== index),
    })),

  updateLeg: (index, leg) =>
    set((state) => ({
      legs: state.legs.map((l, i) => (i === index ? leg : l)),
    })),

  setLegs: (legs) => set({ legs }),
  setAnalysis: (analysis) => set({ analysis }),
  setTimeSlider: (value) => set({ timeSliderValue: value }),
  setIsAnalyzing: (v) => set({ isAnalyzing: v }),
  clearStrategy: () => set({ legs: [], analysis: null, timeSliderValue: 1 }),
}));
