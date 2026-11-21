// A one-dimensional (scalar) Kalman filter.
//
// The model is the standard linear-Gaussian state-space model restricted to a
// single state variable:
//
//   x[k] = A * x[k-1] + w[k-1],   w ~ N(0, Q)
//   z[k] = H * x[k]     + v[k],   v ~ N(0, R)
//
// where x is the hidden scalar state, z is the scalar measurement, A is the
// state-transition scalar, H is the observation scalar, Q is the process-noise
// variance and R is the measurement-noise variance.
//
// The predict/update recursion is written in the numerically stable
// covariance form (Joseph-free, since we are scalar) rather than the
// information form. For a scalar process the covariance form is both simpler
// and immune to the matrix-inversion pitfalls that plague the general case.

/**
 * A scalar Kalman filter.
 *
 * All variances must be strictly positive. The filter does not attempt to
 * recover from a non-positive variance: a zero or negative variance would make
 * the Kalman gain undefined or destabilising, so the constructor rejects it
 * loudly rather than producing silent NaN downstream.
 */
export class ScalarKalmanFilter {
  /**
   * @param {object} [options]
   * @param {number} [options.processVariance=1]   Process-noise variance Q.
   * @param {number} [options.measurementVariance=1] Measurement-noise variance R.
   * @param {number} [options.transition=1]        State-transition scalar A.
   * @param {number} [options.observation=1]       Observation scalar H.
   * @param {number} [options.estimate=0]          Initial state estimate x0.
   * @param {number} [options.estimateVariance=1]  Initial estimate variance P0.
   */
  constructor(options = {}) {
    const {
      processVariance = 1,
      measurementVariance = 1,
      transition = 1,
      observation = 1,
      estimate = 0,
      estimateVariance = 1,
    } = options;

    // Variances are the only quantities that must be positive for the filter
    // to be well-defined. A and H may legitimately be zero or negative (a
    // negative A models an alternating-sign process), so we do not constrain
    // them beyond finiteness.
    assertFinite(processVariance, "processVariance");
    assertFinite(measurementVariance, "measurementVariance");
    assertFinite(transition, "transition");
    assertFinite(observation, "observation");
    assertFinite(estimate, "estimate");
    assertFinite(estimateVariance, "estimateVariance");

    if (processVariance <= 0) {
      throw new RangeError("processVariance must be > 0");
    }
    if (measurementVariance <= 0) {
      throw new RangeError("measurementVariance must be > 0");
    }
    if (estimateVariance <= 0) {
      throw new RangeError("estimateVariance must be > 0");
    }

    this._processVariance = processVariance;
    this._measurementVariance = measurementVariance;
    this._transition = transition;
    this._observation = observation;
    this._estimate = estimate;
    this._estimateVariance = estimateVariance;
  }

  /** Current process-noise variance Q. */
  get processVariance() {
    return this._processVariance;
  }

  /** Current measurement-noise variance R. */
  get measurementVariance() {
    return this._measurementVariance;
  }

  /** Current state-transition scalar A. */
  get transition() {
    return this._transition;
  }

  /** Current observation scalar H. */
  get observation() {
    return this._observation;
  }

  /** Current state estimate. */
  get estimate() {
    return this._estimate;
  }

  /** Current estimate variance P. */
  get estimateVariance() {
    return this._estimateVariance;
  }

  /**
   * Advance the state estimate one step without incorporating a measurement.
   *
   *   x' = A * x
   *   P' = A * P * A + Q
   *
   * @returns {number} The predicted estimate.
   */
  predict() {
    this._estimate = this._transition * this._estimate;
    this._estimateVariance =
      this._transition * this._estimateVariance * this._transition +
      this._processVariance;
    return this._estimate;
  }

  /**
   * Fold a measurement into the current estimate.
   *
   *   y  = z - H * x          (innovation)
   *   S  = H * P * H + R      (innovation variance)
   *   K  = P * H / S          (Kalman gain)
   *   x' = x + K * y
   *   P' = (1 - K * H) * P
   *
   * The covariance update (1 - K*H)*P is the textbook form. For a scalar
   * system it cannot lose positive-definiteness unless K*H > 1, which only
   * happens if R were negative. R is validated at construction, so the
   * subtraction is safe here.
   *
   * @param {number} measurement The observed scalar value z.
   * @returns {number} The updated estimate.
   */
  update(measurement) {
    assertFinite(measurement, "measurement");
    const innovation = measurement - this._observation * this._estimate;
    const innovationVariance =
      this._observation * this._estimateVariance * this._observation +
      this._measurementVariance;
    const gain = (this._estimateVariance * this._observation) / innovationVariance;
    this._estimate = this._estimate + gain * innovation;
    this._estimateVariance = (1 - gain * this._observation) * this._estimateVariance;
    return this._estimate;
  }

  /**
   * Convenience: predict then update with a single measurement.
   *
   * This is the ordering used by the recursive estimator: the measurement at
   * step k is assumed to have been produced after the process advanced from
   * step k-1 to step k.
   *
   * @param {number} measurement The observed scalar value z.
   * @returns {number} The updated estimate.
   */
  step(measurement) {
    this.predict();
    return this.update(measurement);
  }

  /**
   * Reset the estimate and its variance, keeping the model parameters.
   *
   * @param {number} [estimate=0]          New state estimate.
   * @param {number} [estimateVariance=1]  New estimate variance.
   * @returns {this}
   */
  reset(estimate = 0, estimateVariance = 1) {
    assertFinite(estimate, "estimate");
    assertFinite(estimateVariance, "estimateVariance");
    if (estimateVariance <= 0) {
      throw new RangeError("estimateVariance must be > 0");
    }
    this._estimate = estimate;
    this._estimateVariance = estimateVariance;
    return this;
  }

  /**
   * Snapshot of the current state.
   *
   * @returns {{estimate: number, estimateVariance: number}}
   */
  state() {
    return {
      estimate: this._estimate,
      estimateVariance: this._estimateVariance,
    };
  }
}

/**
 * Factory mirror of the constructor. Provided because callers that build
 * filters from configuration objects often prefer a function to `new`.
 *
 * @param {object} [options] Same options as {@link ScalarKalmanFilter}.
 * @returns {ScalarKalmanFilter}
 */
export function createScalarKalmanFilter(options = {}) {
  return new ScalarKalmanFilter(options);
}

/**
 * Reject NaN and Infinity early. A NaN that enters the recursion silently
 * poisons every subsequent estimate, and the failure surfaces far from its
 * cause. Throwing at the boundary keeps the blame local.
 *
 * @param {unknown} value
 * @param {string} name
 */
function assertFinite(value, name) {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    throw new TypeError(`${name} must be a finite number`);
  }
}
