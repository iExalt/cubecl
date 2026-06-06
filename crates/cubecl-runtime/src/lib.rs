#![no_std]
#![warn(missing_docs)]

//! `CubeCL` runtime crate that helps creating high performance async runtimes.

#[cfg(feature = "std")]
extern crate std;

extern crate alloc;

#[macro_use]
extern crate derive_new;

/// Various identifier types used in `CubeCL`.
pub mod id;

/// Kernel related traits.
pub mod kernel;

/// Stream related utilities.
pub mod stream;

/// Compute client module.
pub mod client;

/// Autotune module
pub mod tune;

/// Memory management module.
pub mod memory_management;
/// Compute server module.
pub mod server;
/// Compute Storage module.
pub mod storage;

/// `CubeCL` config module.
pub mod config;

/// Process-wide cache metrics for startup observability.
pub mod cache_metrics;

/// Process-wide GPU operation counts for deterministic workload observability.
pub mod op_metrics;

pub use cubecl_common::benchmark;

/// Logging utilities to be used by a compute server.
pub mod logging;

/// TMA-related runtime types
pub mod tma;

/// Compiler trait and related types
pub mod compiler;
/// Runtime trait and related types
pub mod runtime;
/// Simple system profiling using timestamps.
pub mod timestamp_profiler;

/// Validation utils for shared properties
pub mod validation;

/// Allocators moddule.
pub mod allocator;
