'use client';

interface LegendEntry {
  value: string;
  color: string;
  payload?: { fillOpacity?: number };
}

export default function ChartLegend({ payload, className }: { payload?: LegendEntry[]; className?: string }) {
  if (!payload) return null;
  return (
    <div className={`flex justify-center gap-6 pt-4 ${className ?? ''}`}>
      {payload.map((entry) => (
        <div key={entry.value} className="flex items-center gap-2">
          <div
            className="w-3 h-3 rounded-sm"
            style={{ backgroundColor: entry.color, opacity: entry.payload?.fillOpacity ?? 1 }}
          />
          <span className="text-sm text-gray-700">{entry.value}</span>
        </div>
      ))}
    </div>
  );
}
