//! Evaluators for automation curves and `AutomationValueSemantics`.

use super::spec::{
    AutomationCurve, AutomationCurves, CurveInterpolation, CurvePoint, FilterMode, SegmentEasing,
    SegmentInterpolation, SpecialResult, SpecialRole, ValueMapping, ValueSemantics, ValueUnit,
};

/// Result of evaluating a raw curve value through [`ValueSemantics`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EvaluatedValue {
    pub numeric: Option<f64>,
    pub unit: ValueUnit,
    pub role: Option<SpecialRole>,
    pub mode: Option<FilterMode>,
}

impl EvaluatedValue {
    /// Convert a numeric gain value (in [`ValueUnit::Linear`], [`ValueUnit::Decibels`],
    /// or [`ValueUnit::Percent`]) to a linear amplitude factor.
    pub fn as_linear_gain(&self) -> Option<f64> {
        let v = self.numeric?;
        Some(match self.unit {
            ValueUnit::Linear | ValueUnit::Hertz => v,
            ValueUnit::Decibels => {
                if v <= -120.0 {
                    0.0
                } else {
                    10.0_f64.powf(v / 20.0)
                }
            }
            ValueUnit::Percent => v / 100.0,
        })
    }
}

/// Evaluate a single [`AutomationCurve`] at normalized position `u in [0.0, 1.0]`.
///
/// Returns `None` if `u` lies outside `[curve.start, curve.end]` or the curve has no points.
pub fn eval_curve(curve: &AutomationCurve, u: f64) -> Option<f64> {
    if curve.points.is_empty() {
        return None;
    }
    let span = curve.end - curve.start;
    if span > 0.0 && (u < curve.start || u > curve.end) {
        return None;
    }

    // Points are typically defined in the curve's x-coordinate (either [0, 1] or [start, end]).
    let x = if curve.points.first().is_some_and(|p| p.x >= curve.start)
        && curve.points.last().is_some_and(|p| p.x <= curve.end)
        && span > 0.0
        && (curve.start > 0.0 || curve.end < 1.0)
    {
        u
    } else if span > 0.0 {
        ((u - curve.start) / span).clamp(0.0, 1.0)
    } else {
        u.clamp(0.0, 1.0)
    };

    if x <= curve.points[0].x {
        return Some(curve.points[0].y);
    }
    let last = curve.points[curve.points.len() - 1];
    if x >= last.x {
        return Some(last.y);
    }

    if curve.interpolation == CurveInterpolation::CubicBezier && curve.points.len() >= 4 {
        return Some(eval_cubic_bezier_chain(&curve.points, x));
    }

    // Binary search for the segment containing x.
    let idx = curve.points.partition_point(|p| p.x < x);
    let p0 = curve.points[idx.saturating_sub(1)];
    let p1 = curve.points[idx];
    let dx = p1.x - p0.x;
    if dx.abs() < f64::EPSILON {
        return Some(p1.y);
    }
    let t = ((x - p0.x) / dx).clamp(0.0, 1.0);
    Some(p0.y + t * (p1.y - p0.y))
}

/// Evaluate a 4-point (or chained `3k + 1`-point) monotonic cubic Bézier curve at coordinate `x`.
///
/// Uses safeguarded Newton-Raphson with bisection fallback to invert `B_x(t) = x`, which remains
/// accurate to `< 1e-12` even when endpoints have zero horizontal derivative (e.g. `P1.x == P0.x`).
fn eval_cubic_bezier_chain(points: &[CurvePoint], x: f64) -> f64 {
    let seg_count = (points.len() - 1) / 3;
    let mut seg_idx = 0;
    for s in 0..seg_count {
        let base = s * 3;
        if x >= points[base].x && x <= points[base + 3].x {
            seg_idx = s;
            break;
        }
        if x > points[base + 3].x {
            seg_idx = s;
        }
    }

    let base = seg_idx * 3;
    let p0 = points[base];
    let p1 = points[base + 1];
    let p2 = points[base + 2];
    let p3 = points[base + 3];

    if x <= p0.x {
        return p0.y;
    }
    if x >= p3.x {
        return p3.y;
    }

    let ax = p3.x - 3.0 * p2.x + 3.0 * p1.x - p0.x;
    let bx = 3.0 * (p2.x - 2.0 * p1.x + p0.x);
    let cx = 3.0 * (p1.x - p0.x);
    let dx = p0.x;

    let mut lo = 0.0_f64;
    let mut hi = 1.0_f64;
    let span_x = (p3.x - p0.x).max(f64::EPSILON);
    let mut t = ((x - p0.x) / span_x).clamp(0.0, 1.0);

    for _ in 0..24 {
        let bx_val = ((ax * t + bx) * t + cx) * t + dx;
        let err = bx_val - x;
        if err.abs() < 1e-12 {
            break;
        }
        if err < 0.0 {
            lo = t;
        } else {
            hi = t;
        }
        let deriv = (3.0 * ax * t + 2.0 * bx) * t + cx;
        let next = if deriv.abs() > 1e-12 {
            let candidate = t - err / deriv;
            if candidate > lo && candidate < hi {
                candidate
            } else {
                0.5 * (lo + hi)
            }
        } else {
            0.5 * (lo + hi)
        };
        t = next;
    }

    let ay = p3.y - 3.0 * p2.y + 3.0 * p1.y - p0.y;
    let by = 3.0 * (p2.y - 2.0 * p1.y + p0.y);
    let cy = 3.0 * (p1.y - p0.y);
    let dy = p0.y;
    ((ay * t + by) * t + cy) * t + dy
}

/// Evaluate an [`AutomationCurves`] collection at normalized transition progress `u in [0.0, 1.0]`.
pub fn eval_curves(curves: &AutomationCurves, u: f64) -> Option<f64> {
    let u = u.clamp(0.0, 1.0);
    let mut matched = None;
    for c in &curves.curves {
        if let Some(val) = eval_curve(c, u) {
            matched = Some(val);
            break;
        }
    }
    let raw = matched.or_else(|| {
        // If `u` falls into a gap between sub-curves, hold the nearest boundary point.
        curves
            .curves
            .iter()
            .filter(|c| !c.points.is_empty())
            .min_by(|a, b| {
                let da = if u < a.start {
                    a.start - u
                } else {
                    (u - a.end).max(0.0)
                };
                let db = if u < b.start {
                    b.start - u
                } else {
                    (u - b.end).max(0.0)
                };
                da.total_cmp(&db)
            })
            .and_then(|c| {
                if u < c.start {
                    c.points.first().map(|p| p.y)
                } else {
                    c.points.last().map(|p| p.y)
                }
            })
    })?;

    let clamped = if curves.maximum > curves.minimum {
        raw.clamp(curves.minimum, curves.maximum)
    } else {
        raw
    };
    Some(clamped)
}

/// Map a raw automation value through [`ValueSemantics`].
pub fn apply_semantics(raw: f64, semantics: &ValueSemantics) -> EvaluatedValue {
    const EPS: f64 = 1e-9;

    let unit = semantics.output_unit;

    for sc in &semantics.special_cases {
        if raw >= sc.raw_min - EPS && raw <= sc.raw_max + EPS {
            return match sc.result {
                SpecialResult::Role(role) => EvaluatedValue {
                    numeric: None,
                    unit,
                    role: Some(role),
                    mode: None,
                },
                SpecialResult::Numeric(val) => EvaluatedValue {
                    numeric: Some(val),
                    unit,
                    role: None,
                    mode: None,
                },
            };
        }
    }

    match &semantics.mapping {
        ValueMapping::Identity => EvaluatedValue {
            numeric: Some(raw),
            unit,
            role: None,
            mode: None,
        },
        ValueMapping::Base10LogRatio {
            raw_reference,
            output_multiplier,
        } => {
            if raw <= 0.0 || *raw_reference <= 0.0 {
                EvaluatedValue {
                    numeric: Some(f64::NEG_INFINITY),
                    unit,
                    role: Some(SpecialRole::Mute),
                    mode: None,
                }
            } else {
                EvaluatedValue {
                    numeric: Some(output_multiplier * (raw / raw_reference).log10()),
                    unit,
                    role: None,
                    mode: None,
                }
            }
        }
        ValueMapping::Piecewise(segments) => {
            for seg in segments {
                let lo = seg.raw_start.min(seg.raw_end);
                let hi = seg.raw_start.max(seg.raw_end);
                if raw >= lo - EPS && raw <= hi + EPS {
                    let span = seg.raw_end - seg.raw_start;
                    let t_raw = if span.abs() < EPS {
                        0.0
                    } else {
                        ((raw - seg.raw_start) / span).clamp(0.0, 1.0)
                    };
                    let t = match seg.easing {
                        SegmentEasing::None => t_raw,
                        SegmentEasing::PowerIn(exp) => t_raw.powf(exp),
                        SegmentEasing::PowerOut(exp) => 1.0 - (1.0 - t_raw).powf(exp),
                    };
                    let out = match seg.interpolation {
                        SegmentInterpolation::Linear => {
                            seg.output_start + t * (seg.output_end - seg.output_start)
                        }
                        SegmentInterpolation::SignedGeometric => {
                            if seg.output_start > 0.0 && seg.output_end > 0.0 {
                                seg.output_start * (seg.output_end / seg.output_start).powf(t)
                            } else if seg.output_start < 0.0 && seg.output_end < 0.0 {
                                -((-seg.output_start)
                                    * ((-seg.output_end) / (-seg.output_start)).powf(t))
                            } else {
                                seg.output_start + t * (seg.output_end - seg.output_start)
                            }
                        }
                    };
                    return EvaluatedValue {
                        numeric: Some(out),
                        unit,
                        role: None,
                        mode: seg.output_mode,
                    };
                }
            }
            EvaluatedValue {
                numeric: Some(raw),
                unit,
                role: None,
                mode: None,
            }
        }
    }
}

/// Default equal-power outgoing gain `sqrt(1 - u)` for `u in [0, 1]`.
#[inline]
pub fn equal_power_out(u: f64) -> f64 {
    (1.0 - u.clamp(0.0, 1.0)).sqrt()
}

/// Default equal-power incoming gain `sqrt(u)` for `u in [0, 1]`.
#[inline]
pub fn equal_power_in(u: f64) -> f64 {
    u.clamp(0.0, 1.0).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mix::spec::{CurvePoint, InterpolationSegment, SpecialCase};

    #[test]
    fn equal_power_gains_preserve_total_power() {
        for i in 0..=100 {
            let u = i as f64 / 100.0;
            let power = equal_power_out(u).powi(2) + equal_power_in(u).powi(2);
            assert!((power - 1.0).abs() < 1e-12);
        }
    }

    #[test]
    fn piecewise_geometric_filter_semantics_maps_lp_bypass_hp() {
        // Bipolar filter knob: [0.0, 0.49] = low-pass 20 Hz..20 kHz, 0.5 = bypass, [0.51, 1.0] = high-pass 20 Hz..20 kHz
        let sem = ValueSemantics {
            output_unit: ValueUnit::Hertz,
            special_cases: vec![SpecialCase {
                raw_min: 0.495,
                raw_max: 0.505,
                result: SpecialResult::Role(SpecialRole::Bypass),
            }],
            mapping: ValueMapping::Piecewise(vec![
                InterpolationSegment {
                    raw_start: 0.0,
                    raw_end: 0.495,
                    output_start: 20.0,
                    output_end: 20_000.0,
                    output_mode: Some(FilterMode::LowPass),
                    interpolation: SegmentInterpolation::SignedGeometric,
                    easing: SegmentEasing::None,
                },
                InterpolationSegment {
                    raw_start: 0.505,
                    raw_end: 1.0,
                    output_start: 20.0,
                    output_end: 20_000.0,
                    output_mode: Some(FilterMode::HighPass),
                    interpolation: SegmentInterpolation::SignedGeometric,
                    easing: SegmentEasing::None,
                },
            ]),
        };

        let center = apply_semantics(0.5, &sem);
        assert_eq!(center.role, Some(SpecialRole::Bypass));

        let mid_lp = apply_semantics(0.2475, &sem);
        assert_eq!(mid_lp.mode, Some(FilterMode::LowPass));
        // Geometric midpoint of 20 Hz and 20,000 Hz is sqrt(400_000) ~= 632.46 Hz
        let hz = mid_lp.numeric.unwrap();
        assert!((hz - (20.0_f64 * 20_000.0).sqrt()).abs() < 1e-6);

        let high_hp = apply_semantics(1.0, &sem);
        assert_eq!(high_hp.mode, Some(FilterMode::HighPass));
        assert!((high_hp.numeric.unwrap() - 20_000.0).abs() < 1e-6);
    }

    #[test]
    fn curves_evaluate_and_clamp() {
        let curves = AutomationCurves {
            curves: vec![AutomationCurve::linear(
                vec![
                    CurvePoint { x: 0.0, y: 1.0 },
                    CurvePoint { x: 0.5, y: 0.5 },
                    CurvePoint { x: 1.0, y: 0.0 },
                ],
                0.0,
                1.0,
            )],
            minimum: 0.0,
            maximum: 1.0,
        };
        assert!((eval_curves(&curves, 0.25).unwrap() - 0.75).abs() < 1e-12);
        assert!((eval_curves(&curves, 0.75).unwrap() - 0.25).abs() < 1e-12);
    }

    #[test]
    fn cubic_bezier_evaluates_degenerate_line_and_smooth_s_curves() {
        // 1. Degenerate straight-line Bézier where P1 == P0 and P2 == P3 (zero endpoint derivative).
        let straight = AutomationCurve::cubic_bezier(
            vec![
                CurvePoint { x: 0.0, y: 0.5 },
                CurvePoint { x: 0.0, y: 0.5 },
                CurvePoint { x: 1.0, y: 1.0 },
                CurvePoint { x: 1.0, y: 1.0 },
            ],
            0.5,
            1.0,
        );
        for i in 0..=20 {
            let u = 0.5 + 0.5 * (i as f64 / 20.0);
            let local_x = (u - 0.5) / 0.5;
            let expected = 0.5 + 0.5 * local_x;
            let actual = eval_curve(&straight, u).unwrap();
            assert!(
                (actual - expected).abs() < 1e-9,
                "u={u}: got {actual}, expected {expected}"
            );
        }

        // 2. Symmetric S-curve with repeated P1.x == P2.x == 0.5 (Spotify bass-swap EQ transition).
        let s_curve = AutomationCurve::cubic_bezier(
            vec![
                CurvePoint { x: 0.0, y: 0.5 },
                CurvePoint { x: 0.5, y: 0.5 },
                CurvePoint { x: 0.5, y: 0.0 },
                CurvePoint { x: 1.0, y: 0.0 },
            ],
            0.0,
            1.0,
        );
        assert!((eval_curve(&s_curve, 0.0).unwrap() - 0.5).abs() < 1e-12);
        assert!((eval_curve(&s_curve, 0.5).unwrap() - 0.25).abs() < 1e-12);
        assert!(eval_curve(&s_curve, 1.0).unwrap().abs() < 1e-12);
        // Must be strictly smooth and monotonic decreasing (not a hard step at x=0.5).
        let q1 = eval_curve(&s_curve, 0.25).unwrap();
        let q3 = eval_curve(&s_curve, 0.75).unwrap();
        assert!(
            q1 < 0.49 && q1 > 0.25,
            "expected smooth drop at q1, got {q1}"
        );
        assert!(
            q3 > 0.01 && q3 < 0.25,
            "expected smooth tail at q3, got {q3}"
        );
        assert!((q1 + q3 - 0.5).abs() < 1e-9);

        // 3. Asymmetric S-curve [(0,1), (0.6,1), (0.6,0), (1,0)] stays monotonic over [0, 1].
        let asym = AutomationCurve::cubic_bezier(
            vec![
                CurvePoint { x: 0.0, y: 1.0 },
                CurvePoint { x: 0.6, y: 1.0 },
                CurvePoint { x: 0.6, y: 0.0 },
                CurvePoint { x: 1.0, y: 0.0 },
            ],
            0.0,
            1.0,
        );
        let mut prev = 1.0_f64;
        for i in 1..=50 {
            let u = i as f64 / 50.0;
            let v = eval_curve(&asym, u).unwrap();
            assert!(v <= prev + 1e-12, "non-monotonic at u={u}: {v} > {prev}");
            prev = v;
        }
        assert!(prev.abs() < 1e-12);
    }

    #[test]
    fn mixed_linear_and_cubic_bezier_segments_stitch_smoothly() {
        let eq_out = AutomationCurves {
            curves: vec![
                AutomationCurve::linear(
                    vec![CurvePoint { x: 0.0, y: 0.5 }, CurvePoint { x: 1.0, y: 0.5 }],
                    0.0,
                    0.49609375,
                ),
                AutomationCurve::cubic_bezier(
                    vec![
                        CurvePoint { x: 0.0, y: 0.5 },
                        CurvePoint { x: 0.5, y: 0.5 },
                        CurvePoint { x: 0.5, y: 0.0 },
                        CurvePoint { x: 1.0, y: 0.0 },
                    ],
                    0.49609375,
                    0.5,
                ),
                AutomationCurve::linear(
                    vec![CurvePoint { x: 0.0, y: 0.0 }, CurvePoint { x: 1.0, y: 0.0 }],
                    0.5,
                    1.0,
                ),
            ],
            minimum: 0.0,
            maximum: 1.0,
        };
        assert!((eval_curves(&eq_out, 0.25).unwrap() - 0.5).abs() < 1e-12);
        let mid = eval_curves(&eq_out, 0.498046875).unwrap();
        assert!((mid - 0.25).abs() < 1e-9, "got {mid}");
        assert!(eval_curves(&eq_out, 0.75).unwrap().abs() < 1e-12);
    }

    #[test]
    fn decibel_and_percent_semantics_convert_to_linear_gains() {
        let db_sem = ValueSemantics {
            output_unit: ValueUnit::Decibels,
            mapping: ValueMapping::Base10LogRatio {
                raw_reference: 1.0,
                output_multiplier: 20.0,
            },
            special_cases: vec![],
        };
        let half = apply_semantics(0.5, &db_sem);
        assert!((half.numeric.unwrap() - (-6.0205999)).abs() < 1e-4);
        assert!((half.as_linear_gain().unwrap() - 0.5).abs() < 1e-12);

        let zero = apply_semantics(0.0, &db_sem);
        assert_eq!(zero.role, Some(SpecialRole::Mute));

        let pct_sem = ValueSemantics {
            output_unit: ValueUnit::Percent,
            mapping: ValueMapping::Identity,
            special_cases: vec![],
        };
        assert!((apply_semantics(50.0, &pct_sem).as_linear_gain().unwrap() - 0.5).abs() < 1e-12);
    }
}
