/**
 * DentalReferenceModel.js — 32-tooth anatomical reference model, half-arch bilateral
 * symmetry, tooth-type spatial classification, and per-tooth error analysis.
 *
 * ============================================================================
 * HUMAN DENTAL ANATOMY & BILATERAL SYMMETRY (32-TOOTH MODEL)
 * ============================================================================
 * A complete permanent human dentition comprises 32 teeth organized across four
 * quadrants (8 teeth per quadrant):
 *
 *   Upper Right (Q1, FDI 11-18)  |  Upper Left (Q2, FDI 21-28)
 *   ------------------------------------------------------------
 *   Lower Right (Q4, FDI 41-48)  |  Lower Left (Q3, FDI 31-38)
 *
 * Because human dental arches exhibit bilateral symmetry, entering dimensions for
 * the 8 teeth of the upper half-arch and 8 teeth of the lower half-arch fully
 * defines the reference dimensions for all 32 individual teeth:
 *
 *   1. Central Incisor (CI / I1)  — mesial, adjacent to dental midline
 *   2. Lateral Incisor (LI / I2)  — distal to central incisor
 *   3. Canine / Cuspid (C)        — corner of the dental arch
 *   4. First Premolar (PM1)       — bicuspid anterior to molars
 *   5. Second Premolar (PM2)      — bicuspid adjacent to first molar
 *   6. First Molar (M1)           — primary chewing table
 *   7. Second Molar (M2)          — 12-year molar
 *   8. Third Molar (M3)           — wisdom tooth
 *
 * Standard defaults are derived from Wheeler's Dental Anatomy, Physiology and
 * Occlusion (Nelson & Ash, 10th Ed.) and can be overridden with patient-specific
 * odontometric records, stone casts, or digital intraoral scans.
 */

export const TOOTH_TYPES = [
  { key: 'CI',  name: 'Central Incisor', group: 'incisor',  pos: 1 },
  { key: 'LI',  name: 'Lateral Incisor', group: 'incisor',  pos: 2 },
  { key: 'C',   name: 'Canine',          group: 'canine',   pos: 3 },
  { key: 'PM1', name: '1st Premolar',    group: 'premolar', pos: 4 },
  { key: 'PM2', name: '2nd Premolar',    group: 'premolar', pos: 5 },
  { key: 'M1',  name: '1st Molar',       group: 'molar',    pos: 6 },
  { key: 'M2',  name: '2nd Molar',       group: 'molar',    pos: 7 },
  { key: 'M3',  name: '3rd Molar',       group: 'molar',    pos: 8 },
];

/** Standard adult morphometric norms in mm (crown width x crown height). */
export const DEFAULT_HALF_ARCH = {
  upper: {
    CI:  { width: 8.5,  height: 10.5 },
    LI:  { width: 6.5,  height: 9.0  },
    C:   { width: 7.5,  height: 10.0 },
    PM1: { width: 7.0,  height: 8.5  },
    PM2: { width: 6.8,  height: 8.0  },
    M1:  { width: 10.0, height: 7.5  },
    M2:  { width: 9.5,  height: 7.0  },
    M3:  { width: 8.5,  height: 6.5  },
  },
  lower: {
    CI:  { width: 5.0,  height: 9.0  },
    LI:  { width: 5.5,  height: 9.5  },
    C:   { width: 7.0,  height: 10.5 },
    PM1: { width: 7.0,  height: 8.5  },
    PM2: { width: 7.0,  height: 8.0  },
    M1:  { width: 10.5, height: 7.5  },
    M2:  { width: 10.0, height: 7.0  },
    M3:  { width: 9.0,  height: 6.5  },
  },
};

/** Palette for segregated tooth-type rendering. */
export const TOOTH_TYPE_COLORS = {
  incisor:  '#00e5ff',  // Bright Cyan
  canine:   '#ffd166',  // Amber / Warm Gold
  premolar: '#b388ff',  // Soft Violet
  molar:    '#00f5d4',  // Emerald Green
  unknown:  '#90caf9',
};

export class DentalReferenceModel {
  /**
   * @param {object} [customDimensions] Optional custom dimensions object structured like DEFAULT_HALF_ARCH.
   */
  constructor(customDimensions = null) {
    this.dimensions = {
      upper: { ...DEFAULT_HALF_ARCH.upper },
      lower: { ...DEFAULT_HALF_ARCH.lower },
    };
    if (customDimensions) {
      this.setDimensions(customDimensions);
    }
  }

  /**
   * Deep clone and set dimensions.
   */
  setDimensions(dims) {
    if (!dims) return;
    if (dims.upper) {
      for (const [k, v] of Object.entries(dims.upper)) {
        if (v && Number.isFinite(v.width) && Number.isFinite(v.height)) {
          this.dimensions.upper[k] = { width: v.width, height: v.height };
        }
      }
    }
    if (dims.lower) {
      for (const [k, v] of Object.entries(dims.lower)) {
        if (v && Number.isFinite(v.width) && Number.isFinite(v.height)) {
          this.dimensions.lower[k] = { width: v.width, height: v.height };
        }
      }
    }
  }

  /** Reset back to textbook anatomical defaults. */
  resetDefaults() {
    this.dimensions = {
      upper: JSON.parse(JSON.stringify(DEFAULT_HALF_ARCH.upper)),
      lower: JSON.parse(JSON.stringify(DEFAULT_HALF_ARCH.lower)),
    };
  }

  /**
   * Retrieve reference dimension for a specific tooth type and arch.
   *
   * @param {'upper'|'lower'} arch
   * @param {string} toothType e.g. 'CI', 'LI', 'C', 'PM1', 'PM2', 'M1', etc.
   * @returns {{width: number, height: number}}
   */
  getReference(arch, toothType) {
    const archKey = arch === 'lower' ? 'lower' : 'upper';
    return this.dimensions[archKey]?.[toothType] ?? this.dimensions[archKey]?.CI ?? { width: 8.0, height: 9.0 };
  }

  /**
   * Expand the half-arch dimensions symmetrically to generate all 32 teeth (FDI notation).
   *
   * @returns {Object<number, {fdi: number, arch: string, side: string, type: string, name: string, width: number, height: number}>}
   */
  getFull32TeethModel() {
    const model = {};
    const quadrants = [
      { q: 1, arch: 'upper', side: 'right' }, // Upper Right FDI 11..18
      { q: 2, arch: 'upper', side: 'left'  }, // Upper Left  FDI 21..28
      { q: 3, arch: 'lower', side: 'left'  }, // Lower Left  FDI 31..38
      { q: 4, arch: 'lower', side: 'right' }, // Lower Right FDI 41..48
    ];

    quadrants.forEach(({ q, arch, side }) => {
      TOOTH_TYPES.forEach((t) => {
        const fdi = q * 10 + t.pos;
        const ref = this.getReference(arch, t.key);
        model[fdi] = {
          fdi,
          quadrant: q,
          arch,
          side,
          typeKey: t.key,
          name: `${arch === 'upper' ? 'Upper' : 'Lower'} ${side === 'right' ? 'Right' : 'Left'} ${t.name}`,
          group: t.group,
          pos: t.pos,
          width: ref.width,
          height: ref.height,
        };
      });
    });

    return model;
  }

  /**
   * Classify detected tooth tracks along an arch based on their horizontal positions
   * relative to the arch midline (u ≈ 0 in mouth-local coordinates).
   *
   * Teeth are arranged along the dental arch from the midline outwards:
   *   [Distal Right ...] [LI] [CI] | Midline (u=0) | [CI] [LI] [... Distal Left]
   *
   * @param {Array<ToothTrack>} tracks All active tracks on this arch
   * @param {'upper'|'lower'} arch
   * @returns {Array<object>} Array of classified tooth objects
   */
  classifyArchTracks(tracks, arch) {
    if (!tracks || !tracks.length) return [];

    // Filter tracks for this arch and sort strictly by horizontal center coordinate u
    const archTracks = tracks
      .filter((t) => (t.arch ?? 'upper') === arch)
      .map((t) => {
        const s = t.smoothed ?? t;
        return { track: t, u: s.center.u, v: s.center.v };
      })
      .sort((a, b) => a.u - b.u);

    if (!archTracks.length) return [];

    // The mouth-local coordinate system sets u=0 at the center of the outer lip contour.
    // However, the visible teeth cluster might have an empirical center of mass.
    // If the span crosses 0, 0 is the midline. Otherwise, the midpoint of the central-most gap.
    let midlineU = 0.0;
    const uCoords = archTracks.map((item) => item.u);
    const minU = uCoords[0];
    const maxU = uCoords[uCoords.length - 1];

    if (minU < 0 && maxU > 0) {
      midlineU = 0.0;
    } else {
      midlineU = (minU + maxU) / 2;
    }

    // Split tracks into subject's right side (u < midlineU) and left side (u >= midlineU)
    // Note: in mouth-local coordinates:
    // +X is towards subject's right mouth corner (positive u), -X is towards subject's left corner.
    // Let's sort by distance from midline outwards.
    const rightSide = [];
    const leftSide = [];

    archTracks.forEach((item) => {
      if (item.u >= midlineU) {
        rightSide.push(item); // increasing distance from midline
      } else {
        leftSide.push(item);  // decreasing u (increasing distance from midline to the left)
      }
    });

    // Sort outwards from midline
    rightSide.sort((a, b) => Math.abs(a.u - midlineU) - Math.abs(b.u - midlineU));
    leftSide.sort((a, b) => Math.abs(a.u - midlineU) - Math.abs(b.u - midlineU));

    const classified = [];

    const assignSide = (sideTracks, sideName, quadrant) => {
      sideTracks.forEach((item, idx) => {
        const toothType = TOOTH_TYPES[Math.min(idx, TOOTH_TYPES.length - 1)];
        const fdi = quadrant * 10 + toothType.pos;
        const ref = this.getReference(arch, toothType.key);

        classified.push({
          track: item.track,
          id: item.track.id,
          arch,
          side: sideName,
          quadrant,
          fdi,
          toothType: toothType.key,
          toothName: toothType.name,
          group: toothType.group,
          color: TOOTH_TYPE_COLORS[toothType.group] ?? TOOTH_TYPE_COLORS.unknown,
          label: `${arch === 'upper' ? 'U' : 'L'}-${toothType.key}`,
          refWidth: ref.width,
          refHeight: ref.height,
          u: item.u,
          v: item.v,
        });
      });
    };

    if (arch === 'upper') {
      assignSide(rightSide, 'right', 1); // Upper Right: Q1
      assignSide(leftSide,  'left',  2); // Upper Left:  Q2
    } else {
      assignSide(leftSide,  'left',  3); // Lower Left:  Q3
      assignSide(rightSide, 'right', 4); // Lower Right: Q4
    }

    return classified;
  }

  /**
   * Classify all visible tracks across both arches.
   *
   * @param {Array<ToothTrack>} tracks
   * @returns {Array<object>}
   */
  classifyAllTracks(tracks) {
    const upper = this.classifyArchTracks(tracks, 'upper');
    const lower = this.classifyArchTracks(tracks, 'lower');
    return [...upper, ...lower];
  }

  /**
   * Compute error metrics for classified teeth by comparing each tooth against its
   * particular tooth type reference.
   *
   * @param {Array<object>} classifiedTeeth List of classified teeth from classifyAllTracks()
   * @param {Array<object>} toothSizesMm [{ id, wMm, hMm }] from camera pipeline
   * @returns {object|null}
   */
  computeParticularErrors(classifiedTeeth, toothSizesMm) {
    if (!classifiedTeeth || !classifiedTeeth.length || !toothSizesMm || !toothSizesMm.length) {
      return null;
    }

    const sizeMap = new Map();
    toothSizesMm.forEach((s) => sizeMap.set(s.id, s));

    const toothErrors = [];
    const groupMap = {};

    classifiedTeeth.forEach((tooth) => {
      const size = sizeMap.get(tooth.id);
      if (!size || !Number.isFinite(size.wMm) || !Number.isFinite(size.hMm)) return;

      const wAbs = Math.abs(size.wMm - tooth.refWidth);
      const hAbs = Math.abs(size.hMm - tooth.refHeight);
      const wRel = tooth.refWidth > 0 ? wAbs / tooth.refWidth : 0;
      const hRel = tooth.refHeight > 0 ? hAbs / tooth.refHeight : 0;

      const item = {
        id: tooth.id,
        arch: tooth.arch,
        side: tooth.side,
        fdi: tooth.fdi,
        type: tooth.toothType,
        group: tooth.group,
        label: tooth.label,
        color: tooth.color,
        measuredW: size.wMm,
        measuredH: size.hMm,
        refW: tooth.refWidth,
        refH: tooth.refHeight,
        wAbs,
        hAbs,
        wRel,
        hRel,
      };

      toothErrors.push(item);

      if (!groupMap[tooth.group]) groupMap[tooth.group] = [];
      groupMap[tooth.group].push(item);
    });

    if (!toothErrors.length) return null;

    const mean = (arr) => arr.reduce((sum, v) => sum + v, 0) / arr.length;

    const overall = {
      n: toothErrors.length,
      widthMAE:  mean(toothErrors.map((e) => e.wAbs)),
      heightMAE: mean(toothErrors.map((e) => e.hAbs)),
      widthMAPE: mean(toothErrors.map((e) => e.wRel)),
      heightMAPE: mean(toothErrors.map((e) => e.hRel)),
    };

    const byGroup = {};
    for (const [grp, items] of Object.entries(groupMap)) {
      byGroup[grp] = {
        n: items.length,
        widthMAE:  mean(items.map((e) => e.wAbs)),
        heightMAE: mean(items.map((e) => e.hAbs)),
        widthMAPE: mean(items.map((e) => e.wRel)),
        heightMAPE: mean(items.map((e) => e.hRel)),
      };
    }

    return {
      overall,
      byGroup,
      teeth: toothErrors,
    };
  }
}
