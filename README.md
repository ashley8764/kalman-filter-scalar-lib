# kalman-filter-scalar-lib

A zero-dependency scalar Kalman filter that recursively estimates a noisy one-dimensional process from a stream of noisy measurements.

```js
import { ScalarKalmanFilter } from "./src/index.js";

const filter = new ScalarKalmanFilter({
  processVariance: 0.01,
  measurementVariance: 0.25,
  estimate: 0,
  estimateVariance: 1,
});

for (const z of [1.1, 1.9, 3.2, 3.8]) {
  const estimate = filter.step(z);
  console.log(estimate);
}
```

## Why this exists

The general Kalman filter is a matrix algorithm, and most libraries that implement it pull in a linear-algebra dependency. For a single state variable that is unnecessary weight: the entire recursion collapses to scalar arithmetic, and the covariance update cannot lose positive-definiteness the way a matrix inverse can. This library is the scalar case written out by hand, with no dependencies at all.

The trade-off is scope. There is exactly one state variable and exactly one measurement per step. If you need to estimate position and velocity together, or fuse several sensors, this is the wrong tool and you should reach for a full matrix implementation.

## Model

The filter implements the standard linear-Gaussian model

```
x[k] = A * x[k-1] + w[k-1],   w ~ N(0, Q)
z[k] = H * x[k]     + v[k],   v ~ N(0, R)
```

where `A` is `transition`, `H` is `observation`, `Q` is `processVariance`, and `R` is `measurementVariance`.

## API

The package exports two names from `src/index.js`:

- `ScalarKalmanFilter` — the class.
- `createScalarKalmanFilter(options)` — a factory that returns `new ScalarKalmanFilter(options)`.

### `new ScalarKalmanFilter(options?)`

All options are optional and default to `1`, except `estimate` which defaults to `0`.

| option | default | meaning |
| --- | --- | --- |
| `processVariance` | `1` | `Q`, must be `> 0` |
| `measurementVariance` | `1` | `R`, must be `> 0` |
| `transition` | `1` | `A`, any finite number |
| `observation` | `1` | `H`, any finite number |
| `estimate` | `0` | initial state estimate `x0` |
| `estimateVariance` | `1` | initial estimate variance `P0`, must be `> 0` |

### Methods

- `predict()` — advances the estimate one step using `A` and `Q`; returns the predicted estimate.
- `update(z)` — folds measurement `z` into the estimate; returns the updated estimate.
- `step(z)` — `predict()` followed by `update(z)`; returns the updated estimate. This is the ordering used when measurement `k` was produced after the process advanced from step `k-1` to step `k`.
- `reset(estimate = 0, estimateVariance = 1)` — restores the estimate and variance without touching the model parameters; returns the filter.
- `state()` — returns a fresh `{ estimate, estimateVariance }` object.

### Properties

Read-only getters: `processVariance`, `measurementVariance`, `transition`, `observation`, `estimate`, `estimateVariance`.

## Edge cases

Variances must be strictly positive. A zero or negative variance makes the Kalman gain undefined or destabilising, so the constructor and `reset` throw a `RangeError` rather than letting `NaN` propagate through the recursion. Non-numeric and non-finite inputs (`NaN`, `Infinity`, strings) throw a `TypeError`. `transition` and `observation` may be any finite number, including zero and negatives, since a negative transition legitimately models an alternating-sign process.

## Tests

```
node --test
```
