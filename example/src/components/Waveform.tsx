import React, { useMemo } from 'react'
import { View } from 'react-native'

const WAVEFORM_BARS = 48

// Static waveform of an audio clip, drawn with plain Views (one bar per
// peak bin) — no native chart dependency needed.
export function Waveform({
  audio,
  color,
  height = 56,
}: {
  audio: Float32Array
  color: string
  height?: number
}) {
  const peaks = useMemo(() => {
    const bins: number[] = new Array(WAVEFORM_BARS).fill(0)
    if (audio.length === 0) return bins
    const step = Math.max(1, Math.floor(audio.length / WAVEFORM_BARS))
    // Stride within each bin so long clips stay cheap to scan
    const stride = Math.max(1, Math.floor(step / 64))
    let maxPeak = 0
    for (let i = 0; i < WAVEFORM_BARS; i += 1) {
      const start = i * step
      const end = Math.min(start + step, audio.length)
      let peak = 0
      for (let j = start; j < end; j += stride) {
        const v = Math.abs(audio[j] ?? 0)
        if (v > peak) peak = v
      }
      bins[i] = peak
      if (peak > maxPeak) maxPeak = peak
    }
    return maxPeak > 0 ? bins.map((p) => p / maxPeak) : bins
  }, [audio])

  return (
    <View
      style={{
        height,
        flexDirection: 'row',
        alignItems: 'center',
        gap: 2,
      }}
    >
      {peaks.map((p, i) => (
        <View
          key={i}
          style={{
            flex: 1,
            borderRadius: 1.5,
            backgroundColor: color,
            height: Math.max(3, p * height),
          }}
        />
      ))}
    </View>
  )
}
