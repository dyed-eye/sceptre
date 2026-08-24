# Geometry: boxes, staircases, and shapes

## Boxes — the base layer

`Box(x1, x2, y1, y2, z1, z2, eps)` is an axis-aligned dielectric brick;
`Structure(waveguide, boxes)` slices the stack into z-uniform segments at
every box start/end (`Structure.segments()`). Later boxes override earlier
ones where they overlap. Box edges are handled by Li's exact rules — for
rectangular obstacles this is the exact geometry, no discretization at all.

## Staircasing curved shapes (and not over-refining)

Any convex cross-section (disk, notched disk, rotated bar) staircases into
K x-strips with the material y-interval computed **exactly at the strip
midpoint**. Two measured facts to save you compute:

- **K converges fast.** On an r = 8.1 mm notched disk at ε = 80, K = 32 is
  geometry-converged (K = 64 gives identical spectra); on an r = 15 mm disk
  the main lines move ≤ 1.1 MHz between K = 32 and K = 64. Strip widths of
  ~0.5–1 mm at these sizes are enough. **Spend compute on the modal
  truncation N, not on K** — the truncation error at high contrast is tens
  of MHz while the staircase error is ~1 MHz.
- The midpoint rule beats area-matching tweaks: exact y-intervals at strip
  midpoints is what all validated results below used.

## The `Shape` layer: level sets + normals

A staircase of boxes is enough for the scalar factorizations, but it has
forgotten where the *boundary* is — and the tensor factorizations
(`"nvf"`, `"kfj"`) need the boundary **normal field**. The `Shape` layer
carries both:

```python
disk = sceptre.Cylinder(cx=0.016, cy=0.017, r=0.015,
                        z1=0.0, z2=0.005, eps=80.0 + 0j, k=64)
struct = sceptre.Structure(wg, shapes=[disk])   # keyword-only argument
```

- `Cylinder` provides an analytic level set (ρ − r), an exact radial
  normal, and an exact-interval staircase identical to the hand recipe
  above. Its `k` (default 64) spans the shape's own x-extent.
- **Subclass `Shape` for anything else**: provide `level_set(x, y)` (signed
  distance, negative inside) and `bbox`; a generic bisection staircase and
  a finite-difference normal come free. Override `normal()` analytically
  when you can — finite differences degenerate on the level-set ridge
  (medial axis) of non-smooth level sets.
- Staircases clamp themselves to the guide, so wall-touching shapes are
  legal (the benchmark disk touches the +y wall).
- Boolean geometry: build the level set directly, e.g. a notched disk is
  `max(phi_disk, -phi_notch)`. One material interval per x-strip is assumed
  by the generic staircase (convex-in-y); override `staircase()` for
  multi-interval sections.

`Structure` accepts boxes and shapes together for the scalar
factorizations. The tensor factorizations require **shapes only** — a box
edge outside every shape window would silently lose Li's inverse rule, so
the combination is rejected with an error rather than degraded (an
axis-aligned `Rectangle` shape is the planned path for box-like geometry
under NVF).

Each z-uniform `Segment` carries the shapes covering its z-interval
(`Segment.shapes`) — that is how the solver routes tensor factorizations
per segment; segments without shapes are uniform background and take the
analytic path.

## Explicit permittivity grids

Boxes and shapes both end up as a `CrossSection` — a rectilinear grid of per-cell
ε. You can also build that grid yourself and pass the resulting `Segment`s to
`SegmentedStructure`, which is what continuously graded (non-binary) media
require; see [inverse-design.md](inverse-design.md) for the convergence payoff
and the constraints.

## Tolerance and Monte-Carlo studies: fix the grid to the guide

When perturbing a geometry (tolerance sweeps, Monte-Carlo atlases), **fix
the discretization grid to the guide, not to the part**: re-fitting strips
to each perturbed sample makes discretization jumps masquerade as
sensitivity. Acceptance test before trusting any sensitivity number:
re-solve the *nominal* geometry with the grid shifted by half a pixel — the
change must sit far below the smallest sensitivity you intend to claim.

## Choosing K for shapes

The `k` default (64) is an absolute strip count across the shape's
x-extent, tuned for cm-scale shapes at X-band. For very large, very small,
or multi-scale shapes, scale it so strips stay near ~1% of the shape span,
and verify once with a K-doubling: if the observable moves, K was too
coarse (measured reference: ≤ 1.1 MHz line shift for 0.94 mm strips on the
r = 15 mm ε = 80 disk).
