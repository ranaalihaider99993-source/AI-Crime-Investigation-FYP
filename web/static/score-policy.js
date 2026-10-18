/* Display-only heuristic. These thresholds are experimental, not calibrated. */
(function (root) {
  const MIN_TOP_SCORE = 0.50;
  const MIN_MARGIN = 0.10;
  function assess(scores) {
    if (!Array.isArray(scores) || scores.length !== 4 ||
        new Set(scores.map(item => item.label)).size !== 4 ||
        scores.some(item => typeof item.label !== 'string' || !Number.isFinite(item.score) || item.score < 0 || item.score > 1) ||
        Math.abs(scores.reduce((sum, item) => sum + item.score, 0) - 1) > 0.00001) {
      throw new Error('Invalid model scores; no result can be displayed.');
    }
    const ranked = [...scores].sort((a, b) => b.score - a.score);
    const margin = ranked[0].score - ranked[1].score;
    const weak = ranked[0].score + 1e-12 < MIN_TOP_SCORE;
    const close = margin + 1e-12 < MIN_MARGIN;
    return {top: ranked[0], margin, weak, close, inconclusive: weak || close};
  }
  const policy = {MIN_TOP_SCORE, MIN_MARGIN, assess};
  root.CrimeScopeScorePolicy = policy;
  if (typeof module !== 'undefined' && module.exports) module.exports = policy;
})(globalThis);
