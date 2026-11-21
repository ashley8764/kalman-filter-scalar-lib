import { test } from "node:test";
import assert from "node:assert/strict";

import { ScalarKalmanFilter, createScalarKalmanFilter } from "../src/index.js";

const EPS = 1e-9;

function close(actual, expected, eps = EPS, message) {
  assert.ok(
    Math.abs(actual - expected) < eps,
    message ?? `expected ${actual} to be within ${eps} of ${expected}`,
  );
}

test("constructor applies defaults", () => {
  const f = new ScalarKalmanFilter();
  assert.equal(f.processVariance, 1);
  assert.equal(f.measurementVariance, 1);
  assert.equal(f.transition, 1);
  assert.equal(f.observation, 1);
  assert.equal(f.estimate, 0);
  assert.equal(f.estimateVariance, 1);
});

test("factory returns a filter equivalent to the constructor", () => {
  const a = createScalarKalmanFilter({ estimate: 3, estimateVariance: 2 });
  const b = new ScalarKalmanFilter({ estimate: 3, estimateVariance: 2 });
  assert.deepEqual(a.state(), b.state());
});

test("predict propagates estimate and variance", () => {
  const f = new ScalarKalmanFilter({
    transition: 2,
    processVariance: 0.5,
    estimate: 3,
    estimateVariance: 4,
  });
  const predicted = f.predict();
  close(predicted, 6);
  close(f.estimate, 6);
  // P' = A*P*A + Q = 2*4*2 + 0.5 = 16.5
  close(f.estimateVariance, 16.5);
});

test("update with a perfect measurement pulls the estimate toward it", () => {
  const f = new ScalarKalmanFilter({
    estimate: 0,
    estimateVariance: 1,
    measurementVariance: 1,
    observation: 1,
  });
  const updated = f.update(2);
  // K = 1/(1+1) = 0.5, x' = 0 + 0.5*2 = 1, P' = 0.5
  close(updated, 1);
  close(f.estimate, 1);
  close(f.estimateVariance, 0.5);
});

test("a very confident prior barely moves", () => {
  const f = new ScalarKalmanFilter({
    estimate: 0,
    estimateVariance: 1e-6,
    measurementVariance: 1,
  });
  f.update(100);
  assert.ok(Math.abs(f.estimate) < 1e-3, `estimate moved too far: ${f.estimate}`);
});

test("a very uncertain prior snaps to the measurement", () => {
  const f = new ScalarKalmanFilter({
    estimate: 0,
    estimateVariance: 1e6,
    measurementVariance: 1,
  });
  f.update(42);
  close(f.estimate, 42, 1e-3);
});

test("repeated updates on a constant signal converge", () => {
  const f = new ScalarKalmanFilter({
    processVariance: 1e-6,
    measurementVariance: 1,
    estimate: 0,
    estimateVariance: 1,
  });
  for (let i = 0; i < 200; i += 1) {
    f.update(5);
  }
  close(f.estimate, 5, 0.05);
});

test("step runs predict then update in that order", () => {
  const f = new ScalarKalmanFilter({
    transition: 1,
    processVariance: 1,
    measurementVariance: 1,
    estimate: 0,
    estimateVariance: 1,
  });
  // predict: x=0, P=2 ; update with z=2: K=2/3, x=4/3, P=2/3
  const stepped = f.step(2);
  close(stepped, 4 / 3);
  close(f.estimateVariance, 2 / 3);
});

test("observation scalar scales the innovation", () => {
  const f = new ScalarKalmanFilter({
    observation: 2,
    estimate: 1,
    estimateVariance: 1,
    measurementVariance: 1,
  });
  // H=2: innovation = z - H*x = 4 - 2 = 2
  // S = H*P*H + R = 4 + 1 = 5 ; K = P*H/S = 2/5
  // x' = 1 + (2/5)*2 = 1.8 ; P' = (1 - (2/5)*2)*1 = 0.2
  f.update(4);
  close(f.estimate, 1.8);
  close(f.estimateVariance, 0.2);
});

test("negative transition is allowed and flips the sign", () => {
  const f = new ScalarKalmanFilter({ transition: -1, estimate: 3, processVariance: 0.5 });
  f.predict();
  close(f.estimate, -3);
});

test("reset restores the estimate and variance", () => {
  const f = new ScalarKalmanFilter({ estimate: 10, estimateVariance: 5 });
  f.update(0);
  f.reset(2, 3);
  assert.deepEqual(f.state(), { estimate: 2, estimateVariance: 3 });
});

test("reset rejects a non-positive variance", () => {
  const f = new ScalarKalmanFilter();
  assert.throws(() => f.reset(0, 0), RangeError);
  assert.throws(() => f.reset(0, -1), RangeError);
});

test("constructor rejects non-positive variances", () => {
  assert.throws(() => new ScalarKalmanFilter({ processVariance: 0 }), RangeError);
  assert.throws(() => new ScalarKalmanFilter({ measurementVariance: -1 }), RangeError);
  assert.throws(() => new ScalarKalmanFilter({ estimateVariance: 0 }), RangeError);
});

test("constructor rejects non-finite numbers", () => {
  assert.throws(() => new ScalarKalmanFilter({ estimate: NaN }), TypeError);
  assert.throws(() => new ScalarKalmanFilter({ transition: Infinity }), TypeError);
  assert.throws(() => new ScalarKalmanFilter({ observation: "1" }), TypeError);
});

test("update rejects a non-finite measurement", () => {
  const f = new ScalarKalmanFilter();
  assert.throws(() => f.update(NaN), TypeError);
  assert.throws(() => f.update(Infinity), TypeError);
  assert.throws(() => f.update(undefined), TypeError);
});

test("state returns a fresh object each call", () => {
  const f = new ScalarKalmanFilter();
  const a = f.state();
  const b = f.state();
  assert.notEqual(a, b);
  assert.deepEqual(a, b);
});

test("a full recursion on a synthetic ramp tracks the signal", () => {
  // Deterministic pseudo-noise: a fixed table, no RNG, no clock.
  const noise = [0.3, -0.2, 0.1, -0.4, 0.2, 0.0, -0.1, 0.4, -0.3, 0.2];
  const f = new ScalarKalmanFilter({
    processVariance: 0.01,
    measurementVariance: 0.25,
    estimate: 0,
    estimateVariance: 1,
  });
  let truth = 0;
  for (let i = 0; i < noise.length; i += 1) {
    truth = i + 1;
    f.step(truth + noise[i]);
  }
  // The filter lags a ramp, but should be in the right neighbourhood.
  assert.ok(
    Math.abs(f.estimate - truth) < 4,
    `estimate ${f.estimate} too far from ${truth}`,
  );
});

test("estimateVariance decreases with repeated measurements when Q is tiny", () => {
  const f = new ScalarKalmanFilter({
    processVariance: 1e-9,
    measurementVariance: 1,
    estimateVariance: 1,
  });
  const before = f.estimateVariance;
  f.update(1);
  const after = f.estimateVariance;
  assert.ok(after < before, `expected ${after} < ${before}`);
});
