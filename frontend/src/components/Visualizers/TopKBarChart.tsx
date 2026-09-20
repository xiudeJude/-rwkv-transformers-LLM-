import React from 'react';
import type { TokenCandidate } from '../../types/schema';

interface TopKBarChartProps {
  candidates: TokenCandidate[];
  nextToken: TokenCandidate;
}

export const TopKBarChart: React.FC<TopKBarChartProps> = ({ candidates, nextToken }) => {
  const maxProb = candidates.length > 0 ? Math.max(...candidates.map((c) => c.prob), 0.01) : 1;

  return (
    <div className="bg-slate-900/60 rounded-xl border border-slate-800/80 p-4 shadow-xl">
      <div className="flex items-center justify-between mb-3 border-b border-slate-800 pb-2">
        <h3 className="text-sm font-semibold text-slate-200 flex items-center gap-2">
          <span>下一 Token 候选分布 (Top-{candidates.length})</span>
        </h3>
        {nextToken && (
          <div className="flex items-center gap-1.5 text-xs">
            <span className="text-slate-400">ArgMax 预测:</span>
            <span className="bg-emerald-500/20 text-emerald-300 border border-emerald-500/30 px-2 py-0.5 rounded font-mono font-bold">
              {nextToken.token.replace(/\n/g, '\\n').replace(/ /g, '␣')}
            </span>
            <span className="text-emerald-400 font-mono">
              {(nextToken.prob * 100).toFixed(1)}%
            </span>
          </div>
        )}
      </div>

      <div className="space-y-2">
        {candidates.map((cand, idx) => {
          const displayToken = cand.token.replace(/\n/g, '\\n').replace(/ /g, '␣');
          const widthPercent = (cand.prob / maxProb) * 100;
          const isTop = idx === 0;

          return (
            <div key={`${cand.id}-${idx}`} className="group flex items-center gap-2 text-xs">
              <span className="w-4 text-right font-mono text-slate-500">{idx + 1}</span>
              <div className="w-24 truncate font-mono text-right font-medium text-slate-300 group-hover:text-white" title={cand.token}>
                {displayToken}
              </div>
              <div className="flex-1 bg-slate-950 h-5 rounded overflow-hidden p-0.5 border border-slate-800">
                <div
                  className={`h-full rounded transition-all duration-300 ${
                    isTop
                      ? 'bg-gradient-to-r from-emerald-600 to-teal-400'
                      : 'bg-gradient-to-r from-indigo-600 to-sky-400'
                  }`}
                  style={{ width: `${Math.max(widthPercent, 1)}%` }}
                />
              </div>
              <span className="w-14 text-right font-mono font-medium text-slate-300">
                {(cand.prob * 100).toFixed(2)}%
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
};
