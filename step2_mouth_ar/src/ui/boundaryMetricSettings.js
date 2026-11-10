import { boundaryOptions } from '../eval/boundaryOptions.js';
let options = boundaryOptions();
const listeners = new Set();
export const getBoundaryOptions = () => ({ ...options });
export const onBoundaryOptions = listener => listeners.add(listener);
export function mountBoundarySettings(root) {
  root.innerHTML = `<h3>Boundary evaluation settings</h3>
    <div class="buttons buttons--wrap"><label>SDF tolerance (pixels) <input data-m="tolerance" type="number" min="0" step="0.1" value="0.1"></label>
    <label>Coverage required to pass (%) <input data-m="coverage" type="number" min="0" max="100" step="1" value="95"></label><button class="btn" data-m="apply">Apply metric settings</button></div>
    <p class="hint">Hausdorff is the largest boundary mismatch (pixels; lower is better). SDF coverage is the proportion of both borders within the tolerance (higher is better), weighted by border length. PASS means coverage meets the required percentage; it does not certify detection quality. Pixels refer to the evaluated image or mouth crop, not screen zoom. Vector distances are sampled at ≤0.05 px with curve flattening at 0.005 px. Settings update image/reference comparisons; rerun batches to apply new settings.</p><p data-m="status" role="status"></p>`;
  root.querySelector('[data-m="apply"]').addEventListener('click', () => {
    try {
      const a=root.querySelector('[data-m="tolerance"]'), b=root.querySelector('[data-m="coverage"]');
      if (!a.value.trim() || !b.value.trim()) throw new Error('Enter both metric settings');
      options=boundaryOptions({tolerancePx:Number(a.value),requiredCoverage:Number(b.value)/100});
      root.querySelector('[data-m="status"]').textContent='Settings applied. Existing batch results keep the settings used when they ran.';
      for (const listener of listeners) listener();
    } catch(e) { root.querySelector('[data-m="status"]').textContent=e.message; }
  });
}
