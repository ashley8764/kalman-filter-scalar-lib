// Public entry point for kalman-filter-scalar-lib.
//
// The library is deliberately tiny: one class, one factory function. Everything
// else lives in core.js so that the package surface stays obvious from the
// README alone.

export { ScalarKalmanFilter, createScalarKalmanFilter } from "./core.js";
