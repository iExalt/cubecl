#[cfg(std_io)]
use alloc::format;
#[cfg(autotune_persistence)]
use alloc::vec::Vec;

#[cfg(autotune_persistence)]
use cubecl_environment::persistence::StoreError;
#[cfg(autotune_persistence)]
use cubecl_environment::persistence::{CacheOption, Namespace, Store, StoreOptions};
#[cfg(autotune_persistence)]
use serde::{Deserialize, Serialize};

use super::{AutotuneError, AutotuneKey, AutotuneOutcome};
use alloc::string::String;
use cubecl_environment::collections::HashMap;

#[derive(Debug)]
pub(crate) enum CacheEntry {
    Done {
        checksum: ChecksumState,
        fastest_index: usize,
    },
    Pending,
}

#[derive(Debug)]
#[allow(dead_code)] // Some variants are not created when the cache isn't saved.
pub(crate) enum ChecksumState {
    Match,
    NoMatch,
    ToBeVerified(String),
}

/// Persistent cache key
#[cfg(autotune_persistence)]
#[derive(Debug, Serialize, Deserialize, PartialEq, Eq, Clone, Hash)]
pub struct PersistentCacheKey<K> {
    /// The autotune key identifying the operation.
    pub key: K,
    checksum: String,
}

/// Persistent cache entry
///
/// Only [`fastest_index`](Self::fastest_index) is read back: hydration seeds the in-memory cache
/// from it and nothing else. Everything below it is stored so a cache entry can be inspected after
/// the fact — why a kernel won, against which measurements, and under which bounds — which is the
/// question that cannot be answered from a live process once tuning is over. That is also why the
/// type is `pub`: reading an entry back is the point.
#[cfg(autotune_persistence)]
#[derive(Debug, Serialize, Deserialize, PartialEq, Eq, Clone)]
pub struct PersistentCacheValue {
    /// Index of the fastest candidate operation.
    pub fastest_index: usize,
    /// Benchmarking results for all autotune candidates.
    pub results: Vec<AutotuneResult>,
    /// Optional input size bounds for which the autotune result applies.
    ///
    /// Defaulted, so entries written before this field existed still decode. Without it every
    /// cached key on every existing installation would fail to read and re-tune from scratch.
    #[serde(default)]
    pub bounds: Option<crate::tune::Bounds>,
    /// Optional execution time limit for the autotune process.
    ///
    /// Defaulted for the same reason as [`bounds`](Self::bounds).
    #[serde(default)]
    pub limit: Option<core::time::Duration>,
}

#[cfg_attr(autotune_persistence, derive(Serialize, Deserialize))]
#[derive(Debug, Clone)]
/// The result of an autotune job.
pub struct AutotuneResult {
    /// The outcome of the benchmark.
    pub outcome: Result<AutotuneOutcome, AutotuneError>,
}

impl AutotuneResult {
    pub(crate) fn error(error: AutotuneError) -> Self {
        Self {
            outcome: Err(error),
        }
    }
    pub(crate) fn success(outcome: AutotuneOutcome) -> Self {
        Self {
            outcome: Ok(outcome),
        }
    }
}

impl Eq for AutotuneResult {}
impl PartialEq for AutotuneResult {
    fn eq(&self, other: &Self) -> bool {
        match (&self.outcome, &other.outcome) {
            (Ok(lhs), Ok(rhs)) => lhs == rhs,
            (Ok(_), Err(_)) => false,
            (Err(_), Ok(_)) => false,
            // We don't have to check the error
            (Err(_), Err(_)) => true,
        }
    }
}

/// Use to find and reuse the best kernel for some input
#[derive(Debug)]
pub(crate) struct TuneCache<K> {
    /// The single in-memory home of tuning state, keyed for the per-launch
    /// lookup: tuned picks, in-flight tunes and checksum verdicts. Hydrated
    /// from the store, which retains nothing itself, and rebuilt when the
    /// environment switches.
    in_memory_cache: HashMap<K, CacheEntry>,
    /// Write-through persistence, or `None` when the persistent cache is
    /// disabled, so no cache file is ever touched. Lazy: entries live in
    /// [`Self::in_memory_cache`] once hydrated, not here.
    #[cfg(autotune_persistence)]
    persistent_cache: Option<Store<PersistentCacheKey<K>, PersistentCacheValue>>,
    /// Legacy JSON entries loaded read-only, when configured. The active store wins.
    #[cfg(std_io)]
    seed_cache: Option<HashMap<K, (String, usize)>>,
    /// Whether everything the store holds has been ingested into
    /// [`Self::in_memory_cache`]. What makes an ordinary miss cost a bool
    /// check rather than a walk; `false` while an asynchronous storage
    /// (browser) is still loading, and again after an environment switch.
    #[cfg(autotune_persistence)]
    hydrated: bool,
    /// The environment generation [`Self::in_memory_cache`] was built under;
    /// see [`cubecl_environment::environment::generation`].
    #[cfg(autotune_persistence)]
    generation: u32,
}

/// Result of the cache try
#[derive(Debug)]
pub enum TuneCacheResult {
    /// An operation is found.
    Hit {
        /// The index of the fastest operation to execute.
        fastest_index: usize,
    },
    /// The operation might be cached, but we don't know yet whether the checksum is valid.
    Unchecked,
    /// A tuning job is in flight for this key — the worker hasn't published a result yet.
    /// Callers that see this fall through to running the operation rather than blocking on
    /// the in-flight job.
    Pending,
    /// No operation is found yet.
    Miss,
}

impl<K: AutotuneKey> TuneCache<K> {
    pub(crate) fn new(
        #[cfg_attr(not(autotune_persistence), allow(unused_variables))] name: &str,
        #[cfg_attr(not(autotune_persistence), allow(unused_variables))] device_id: &str,
    ) -> Self {
        #[cfg(autotune_persistence)]
        {
            use crate::config::RuntimeConfig;
            use alloc::format;

            let config = crate::config::CubeClRuntimeConfig::get();

            if config.autotune.disable_cache {
                return TuneCache {
                    in_memory_cache: HashMap::new(),
                    persistent_cache: None,
                    #[cfg(std_io)]
                    seed_cache: None,
                    hydrated: true,
                    generation: cubecl_environment::environment::generation(),
                };
            }

            // Sampled before the store opens, so a switch landing in between
            // reads as "rebuild", never as "this state belongs to the new
            // environment".
            let generation = cubecl_environment::environment::generation();
            let namespace = Namespace::scoped("autotune", format!("{device_id}/{name}"));
            let mut cache = TuneCache {
                in_memory_cache: HashMap::new(),
                persistent_cache: Some(Store::new(
                    StoreOptions::new()
                        .storage(namespace)
                        .cache(CacheOption::Lazy),
                )),
                #[cfg(std_io)]
                seed_cache: Self::load_seed(name, device_id),
                hydrated: false,
                generation,
            };
            log::info!("Load autotune cache ...");
            cache.load_seed_into_memory();
            let loaded = cache.sync_persistent();
            log::info!("Loaded {loaded} autotune cached entries");

            cache
        }

        #[cfg(not(autotune_persistence))]
        {
            TuneCache {
                in_memory_cache: HashMap::new(),
            }
        }
    }

    #[cfg(std_io)]
    fn load_seed(name: &str, device_id: &str) -> Option<HashMap<K, (String, usize)>> {
        use crate::config::RuntimeConfig;

        let root = crate::config::CubeClRuntimeConfig::get()
            .autotune
            .seed_cache
            .as_ref()?
            .root();
        Self::load_seed_from_path(&Self::legacy_seed_path(&root, name, device_id))
    }

    #[cfg(std_io)]
    fn legacy_seed_path(root: &std::path::Path, name: &str, device_id: &str) -> std::path::PathBuf {
        let mut file = root.join("autotune").join(env!("CARGO_PKG_VERSION"));
        // The old cache accepted a relative `device_id/name` path and sanitized each component.
        // Keep that component boundary: a slash in an identity was a nested legacy directory.
        let path_partial = std::path::PathBuf::from(format!("{device_id}/{name}"));
        for segment in path_partial.iter() {
            // `Path::iter` yields the root component for an absolute identity; the legacy helper
            // skipped that component before sanitizing the remaining path segments.
            if segment == std::ffi::OsStr::new("/") {
                continue;
            }
            let segment = segment.to_string_lossy();
            file.push(sanitize_filename::sanitize_with_options(
                segment,
                sanitize_filename::Options {
                    replacement: "_",
                    ..Default::default()
                },
            ));
        }
        file.set_extension("json.log");
        file
    }

    #[cfg(std_io)]
    fn load_seed_from_path(path: &std::path::Path) -> Option<HashMap<K, (String, usize)>> {
        use serde::Deserialize;

        // This is intentionally a read rather than Cache::new: loading a readonly seed must not
        // create an empty legacy file or its parent directories.
        let bytes = std::fs::read(path).ok()?;
        #[derive(Deserialize)]
        struct Entry<K> {
            key: PersistentCacheKey<K>,
            value: PersistentCacheValue,
        }
        let mut entries = HashMap::new();
        for line in bytes.split(|byte| *byte == b'\n') {
            if line.is_empty() {
                continue;
            }
            if let Ok(entry) = serde_json::from_slice::<Entry<K>>(line) {
                entries.insert(
                    entry.key.key,
                    (entry.key.checksum, entry.value.fastest_index),
                );
            }
        }
        Some(entries)
    }

    #[cfg(std_io)]
    fn load_seed_into_memory(&mut self) {
        let Some(seed_cache) = self.seed_cache.clone() else {
            return;
        };
        let loaded = seed_cache.len() as u64;
        for (key, (checksum, fastest_index)) in seed_cache {
            self.merge_hydrated(key.clone(), checksum.clone(), fastest_index);
        }
        crate::cache_metrics::record_autotune_seed_entries_loaded(loaded);
    }

    /// Merge a hydrated entry without clobbering state produced by this process. A writable
    /// entry is allowed to replace a seed (both are unverified), while a live tune, a locally
    /// finalized result, or a checksum failure remains authoritative until the next reset.
    #[cfg(autotune_persistence)]
    fn merge_hydrated(&mut self, key: K, checksum: String, fastest_index: usize) {
        let replace = match self.in_memory_cache.get(&key) {
            None => true,
            Some(CacheEntry::Pending) => false,
            Some(CacheEntry::Done {
                checksum: ChecksumState::Match | ChecksumState::NoMatch,
                ..
            }) => false,
            Some(CacheEntry::Done {
                checksum: ChecksumState::ToBeVerified(_),
                ..
            }) => true,
        };

        if replace {
            self.in_memory_cache.insert(
                key,
                CacheEntry::Done {
                    checksum: ChecksumState::ToBeVerified(checksum),
                    fastest_index,
                },
            );
        }
    }

    pub fn fastest(&self, key: &K) -> TuneCacheResult {
        let Some(val) = self.in_memory_cache.get(key) else {
            return TuneCacheResult::Miss;
        };

        let CacheEntry::Done {
            checksum,
            fastest_index,
        } = val
        else {
            // Pending: clone the receiver so the caller can subscribe to the in-flight tune.
            let CacheEntry::Pending = val else {
                unreachable!()
            };
            return TuneCacheResult::Pending;
        };

        if cfg!(autotune_persistence) {
            match checksum {
                ChecksumState::ToBeVerified(..) => TuneCacheResult::Unchecked, // Don't know yet.
                ChecksumState::NoMatch => TuneCacheResult::Miss,               // Can't use this.
                ChecksumState::Match => TuneCacheResult::Hit {
                    fastest_index: *fastest_index,
                },
            }
        } else {
            // Clippy;
            let _ = checksum;
            TuneCacheResult::Hit {
                fastest_index: *fastest_index,
            }
        }
    }

    #[cfg(autotune_persistence)]
    pub fn validate_checksum(&mut self, key: &K, checksum: &str) -> TuneCacheResult {
        let Some(val) = self.in_memory_cache.get_mut(key) else {
            return TuneCacheResult::Miss;
        };

        if let CacheEntry::Done {
            checksum: checksum_state,
            ..
        } = val
            && let ChecksumState::ToBeVerified(checksum_expected) = checksum_state
        {
            if checksum_expected == checksum {
                *checksum_state = ChecksumState::Match;
            } else {
                *checksum_state = ChecksumState::NoMatch;
            }
        }

        self.fastest(key)
    }

    /// Mark a key as being tuned. Used by [`Tuner::tune`] under the cache mutex so that
    /// concurrent callers see [`TuneCacheResult::Pending`] instead of starting a second job
    /// for the same key.
    pub(crate) fn mark_pending(&mut self, key: K) {
        self.in_memory_cache.insert(key, CacheEntry::Pending);
    }

    pub(crate) fn cache_insert(&mut self, key: K, fastest_index: usize) {
        self.in_memory_cache.insert(
            key,
            CacheEntry::Done {
                checksum: ChecksumState::Match,
                fastest_index,
            },
        );
    }
}

#[cfg(all(test, std_io, autotune_persistence))]
mod tests {
    use super::{CacheEntry, ChecksumState, TuneCache};
    use crate::tune::TuneCacheResult;
    use alloc::string::{String, ToString};
    use alloc::vec;
    use cubecl_environment::collections::HashMap;
    use cubecl_environment::persistence::{CacheOption, Namespace, Store, StoreOptions};
    use std::io::Write;

    fn cache_without_store() -> TuneCache<String> {
        TuneCache {
            in_memory_cache: HashMap::new(),
            persistent_cache: None,
            seed_cache: Some(HashMap::new()),
            hydrated: true,
            generation: cubecl_environment::environment::generation(),
        }
    }

    fn cache_with_store() -> TuneCache<String> {
        TuneCache {
            in_memory_cache: HashMap::new(),
            persistent_cache: Some(Store::new(
                StoreOptions::new()
                    .storage(Namespace::scoped("autotune-seed-tests", "state"))
                    .cache(CacheOption::Lazy),
            )),
            seed_cache: Some(HashMap::new()),
            hydrated: true,
            generation: cubecl_environment::environment::generation(),
        }
    }

    struct EnvironmentGuard(String);

    impl Drop for EnvironmentGuard {
        fn drop(&mut self) {
            cubecl_environment::environment::activate(&self.0);
        }
    }

    #[test]
    fn legacy_path_uses_cubecl_filename_sanitization() {
        let root = std::path::Path::new("/tmp/cubecl-seed-root");
        let path = TuneCache::<String>::legacy_seed_path(root, "tuner name:1", "device name/1");
        let expected = root
            .join("autotune")
            .join(env!("CARGO_PKG_VERSION"))
            .join("device name")
            .join("1")
            .join("tuner name_1.json.log");

        assert_eq!(path, expected);

        let absolute =
            TuneCache::<String>::legacy_seed_path(root, "tuner name:1", "/device name/1");
        assert_eq!(absolute, expected);
    }

    #[test]
    fn missing_seed_is_read_only_and_legacy_ndjson_decodes() {
        let root = tempfile::tempdir().unwrap();
        let path = TuneCache::<String>::legacy_seed_path(root.path(), "tuner name", "device name");
        assert!(TuneCache::<String>::load_seed_from_path(&path).is_none());
        assert!(!path.exists());
        assert!(!root.path().join("autotune").exists());

        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        let mut file = std::fs::File::create(&path).unwrap();
        // Omit fields added after the legacy format; serde defaults make old seeds decode.
        let legacy_entry = serde_json::json!({
            "key": {"key": "operation", "checksum": "legacy-checksum"},
            "value": {"fastest_index": 4, "results": []}
        });
        serde_json::to_writer(&mut file, &legacy_entry).unwrap();
        writeln!(file).unwrap();
        let before = std::fs::read(&path).unwrap();

        let entries = TuneCache::<String>::load_seed_from_path(&path).unwrap();
        assert_eq!(
            entries.get("operation"),
            Some(&("legacy-checksum".to_string(), 4))
        );
        assert_eq!(std::fs::read(&path).unwrap(), before);
    }

    #[test]
    fn writable_hydration_replaces_seed_but_preserves_local_states() {
        let mut cache = cache_without_store();
        cache
            .seed_cache
            .as_mut()
            .unwrap()
            .insert("operation".to_string(), ("seed-checksum".to_string(), 1));
        cache.load_seed_into_memory();
        assert!(matches!(
            cache.fastest(&"operation".to_string()),
            TuneCacheResult::Unchecked
        ));

        cache.merge_hydrated("operation".to_string(), "writable-checksum".to_string(), 2);
        assert!(matches!(
            cache.fastest(&"operation".to_string()),
            TuneCacheResult::Unchecked
        ));
        assert!(matches!(
            cache.validate_checksum(&"operation".to_string(), "writable-checksum"),
            TuneCacheResult::Hit { fastest_index: 2 }
        ));

        cache.mark_pending("pending".to_string());
        cache.merge_hydrated("pending".to_string(), "stale-checksum".to_string(), 3);
        assert!(matches!(
            cache.fastest(&"pending".to_string()),
            TuneCacheResult::Pending
        ));

        cache.cache_insert("final".to_string(), 5);
        cache.merge_hydrated("final".to_string(), "stale-checksum".to_string(), 6);
        assert!(matches!(
            cache.fastest(&"final".to_string()),
            TuneCacheResult::Hit { fastest_index: 5 }
        ));
    }

    #[test]
    fn invalidated_entries_are_not_replaced_by_stale_hydration() {
        let mut cache = cache_without_store();
        cache.merge_hydrated("operation".to_string(), "old-checksum".to_string(), 1);
        assert!(matches!(
            cache.validate_checksum(&"operation".to_string(), "new-checksum"),
            TuneCacheResult::Miss
        ));

        cache.merge_hydrated("operation".to_string(), "old-checksum".to_string(), 1);
        assert!(matches!(
            cache.fastest(&"operation".to_string()),
            TuneCacheResult::Miss
        ));
        assert!(matches!(
            cache.in_memory_cache.get("operation"),
            Some(CacheEntry::Done {
                checksum: ChecksumState::NoMatch,
                ..
            })
        ));
    }

    #[test]
    #[serial_test::serial]
    fn seed_is_reloaded_after_environment_switch() {
        let _restore = EnvironmentGuard(cubecl_environment::environment::active().to_string());
        cubecl_environment::environment::activate("seed-cache-test-before");
        let mut cache = cache_with_store();
        cache
            .seed_cache
            .as_mut()
            .unwrap()
            .insert("operation".to_string(), ("seed-checksum".to_string(), 7));
        cache.load_seed_into_memory();
        cubecl_environment::environment::activate("seed-cache-test-after");
        cache.reset_if_environment_switched();

        assert!(matches!(
            cache.fastest(&"operation".to_string()),
            TuneCacheResult::Unchecked
        ));
    }
}

#[cfg(autotune_persistence)]
impl<K: AutotuneKey> TuneCache<K> {
    /// Drops tuning state belonging to a previous environment, so a switch
    /// re-hydrates and re-tunes rather than serving the old environment's
    /// picks. One relaxed atomic load when nothing switched.
    ///
    /// In-flight tunes are dropped with everything else: their completion
    /// still records a hardware-valid result, so the whole cost of the race
    /// is one duplicate tune per switch.
    pub(crate) fn reset_if_environment_switched(&mut self) {
        // Persistence disabled means the tuning state is process-local and
        // unbound, like a store without a storage: it survives switches.
        if self.persistent_cache.is_none() {
            return;
        }

        let generation = cubecl_environment::environment::generation();
        if generation == self.generation {
            return;
        }

        log::debug!("Environment switched, resetting the autotune cache");
        self.generation = generation;
        self.in_memory_cache.clear();
        self.hydrated = false;
        #[cfg(std_io)]
        self.load_seed_into_memory();
    }

    /// Ingest everything the persistent store holds into the in-memory cache,
    /// as unverified entries.
    ///
    /// Runs at construction, and again whenever `hydrated` fell back to
    /// `false`: after an environment switch, and on the browser backend while
    /// its asynchronous hydration is still in flight. Once hydrated, a miss
    /// costs one bool check here — never a walk, and never a rescan of the
    /// database under the tuner mutex.
    ///
    /// Returns how many entries the store delivered.
    pub(crate) fn sync_persistent(&mut self) -> usize {
        if self.hydrated {
            return 0;
        }

        let mut delivered = 0usize;
        let mut hydrated = Vec::new();
        let complete = {
            let Some(persistent_cache) = self.persistent_cache.as_mut() else {
                return 0;
            };
            persistent_cache.scan(|key, value| {
                delivered += 1;
                hydrated.push((key.key, key.checksum, value.fastest_index));
            })
        };
        for (key, checksum, fastest_index) in hydrated {
            self.merge_hydrated(key, checksum, fastest_index);
        }
        self.hydrated = complete;
        crate::cache_metrics::record_autotune_writable_entries_loaded(delivered as u64);

        delivered
    }

    pub(crate) fn persistent_cache_insert(
        &mut self,
        key: K,
        checksum: String,
        value: PersistentCacheValue,
    ) {
        let Some(persistent_cache) = self.persistent_cache.as_mut() else {
            return;
        };

        if let Err(err) = persistent_cache.insert(PersistentCacheKey { key, checksum }, value) {
            match err {
                StoreError::DuplicatedKey {
                    key,
                    value_previous,
                    value_updated,
                } => log::warn!(
                    "Autotune the same function multiple times for key {key:?} => old {value_previous:?}, new {value_updated:?}"
                ),
                // Another process sharing the cache root tuned this key first.
                // Routine with N training processes on a cold cache, and both
                // results are valid, so it stays quiet: warning here would
                // print a full result payload per key on every cold start.
                StoreError::KeyOutOfSync { key, .. } => {
                    log::debug!("Autotune result for key {key:?} was already stored concurrently")
                }
                StoreError::Backend { key, error } => log::warn!(
                    "Autotune result for key {key:?} could not be stored, it will be retuned: {error}"
                ),
            }
        } else {
            crate::cache_metrics::record_autotune_write();
        }
    }
}
