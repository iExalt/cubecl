#[cfg(std_io)]
use std::vec::Vec;

#[cfg(std_io)]
use cubecl_common::cache::Cache;
#[cfg(std_io)]
use cubecl_common::cache::CacheError;
#[cfg(std_io)]
use serde::{Deserialize, Serialize};

use super::{AutotuneError, AutotuneKey, AutotuneOutcome};
use alloc::string::String;
use hashbrown::HashMap;

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
#[cfg(std_io)]
#[derive(Debug, Serialize, Deserialize, PartialEq, Eq, Clone, Hash)]
pub(crate) struct PersistentCacheKey<K> {
    key: K,
    checksum: String,
}

/// Persistent cache entry
#[cfg(std_io)]
#[derive(Debug, Serialize, Deserialize, PartialEq, Eq, Clone)]
pub(crate) struct PersistentCacheValue {
    fastest_index: usize,
    results: Vec<AutotuneResult>,
}

#[cfg_attr(std_io, derive(Serialize, Deserialize))]
#[derive(Debug, Clone)]
/// The result of an autotune job.
pub struct AutotuneResult {
    pub(crate) outcome: Result<AutotuneOutcome, AutotuneError>,
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
    in_memory_cache: HashMap<K, CacheEntry>,
    #[cfg(std_io)]
    seed_cache: Option<Cache<PersistentCacheKey<K>, PersistentCacheValue>>,
    #[cfg(std_io)]
    persistent_cache: Cache<PersistentCacheKey<K>, PersistentCacheValue>,
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
    /// The receiver wakes (with `Err(RecvError)`) when the worker commits the result. Native
    /// callers `block_on` it and re-query; wasm callers drop it and fall back.
    Pending,
    /// No operation is found yet.
    Miss,
}

impl<K: AutotuneKey> TuneCache<K> {
    pub(crate) fn new(
        #[cfg_attr(not(std_io), allow(unused_variables))] name: &str,
        #[cfg_attr(not(std_io), allow(unused_variables))] device_id: &str,
    ) -> Self {
        #[cfg(std_io)]
        {
            use crate::config::RuntimeConfig;

            let config = crate::config::CubeClRuntimeConfig::get();
            let root = config.autotune.cache.root();
            let seed_root = config
                .autotune
                .seed_cache
                .as_ref()
                .map(|cache| cache.root());
            Self::new_with_roots(name, device_id, root, seed_root)
        }

        #[cfg(not(std_io))]
        {
            TuneCache {
                in_memory_cache: HashMap::new(),
            }
        }
    }

    #[cfg(std_io)]
    fn new_with_roots(
        name: &str,
        device_id: &str,
        root: std::path::PathBuf,
        seed_root: Option<std::path::PathBuf>,
    ) -> Self {
        use std::format;

        let options = cubecl_common::cache::CacheOption::default();
        let mut cache = TuneCache {
            in_memory_cache: HashMap::new(),
            seed_cache: seed_root.map(|root| {
                Cache::new(
                    format!("{device_id}/{name}"),
                    options.clone().root(root).name("autotune"),
                )
            }),
            persistent_cache: Cache::new(
                format!("{device_id}/{name}"),
                options.root(root).name("autotune"),
            ),
        };
        cache.load();
        cache
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

        if cfg!(std_io) {
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

    #[cfg(std_io)]
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
    /// concurrent callers see [`TuneCacheResult::Pending`] and wait on the same job instead of
    /// starting a second one. Returns `(Sender, Receiver)`:
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

#[cfg(std_io)]
impl<K: AutotuneKey> TuneCache<K> {
    pub(crate) fn persistent_cache_insert(
        &mut self,
        key: K,
        checksum: String,
        fastest_index: usize,
        results: Vec<AutotuneResult>,
    ) {
        if let Err(err) = self.persistent_cache.insert(
            PersistentCacheKey { key, checksum },
            PersistentCacheValue {
                fastest_index,
                results,
            },
        ) {
            match err {
                CacheError::DuplicatedKey {
                    key,
                    value_previous,
                    value_updated,
                } => {
                    log::warn!(
                        "Autotune the same function multiple times for key {key:?} => old {value_previous:?}, new {value_updated:?}"
                    );
                }
                CacheError::KeyOutOfSync { .. } => {
                    // This is OK.
                }
            }
        }
        crate::cache_metrics::record_autotune_write();
        // .expect();
    }

    /// Load the persistent cache data from disk
    pub(crate) fn load(&mut self) {
        let mut seeded = 0;
        if let Some(seed_cache) = self.seed_cache.as_mut() {
            seed_cache.for_each(|key, value| {
                seeded += 1;
                self.in_memory_cache.insert(
                    key.key.clone(),
                    CacheEntry::Done {
                        checksum: ChecksumState::ToBeVerified(key.checksum.clone()),
                        fastest_index: value.fastest_index,
                    },
                );
            });
        }

        let mut writable = 0;
        self.persistent_cache.for_each(|key, value| {
            writable += 1;
            self.in_memory_cache.insert(
                key.key.clone(),
                CacheEntry::Done {
                    checksum: ChecksumState::ToBeVerified(key.checksum.clone()),
                    fastest_index: value.fastest_index,
                },
            );
        });
        crate::cache_metrics::record_autotune_seed_entries_loaded(seeded);
        crate::cache_metrics::record_autotune_writable_entries_loaded(writable);
        log::info!("Loaded {seeded} seeded and {writable} writable autotune cached entries");
    }
}

#[cfg(all(test, std_io))]
mod tests {
    use super::{
        AutotuneResult, PersistentCacheKey, PersistentCacheValue, TuneCache, TuneCacheResult,
    };
    use cubecl_common::cache::{Cache, CacheOption};
    use std::borrow::ToOwned;
    use std::format;
    use std::fs::remove_dir_all;
    use std::path::{Path, PathBuf};
    use std::string::String;
    use std::time::{SystemTime, UNIX_EPOCH};
    use std::vec::Vec;

    fn temp_root(label: &str) -> PathBuf {
        let suffix = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("System time should be after the Unix epoch")
            .as_nanos();
        std::env::temp_dir().join(format!(
            "cubecl-tune-cache-{label}-{}-{suffix}",
            std::process::id()
        ))
    }

    fn persistent_cache(root: &Path) -> Cache<PersistentCacheKey<String>, PersistentCacheValue> {
        Cache::new(
            "device/test",
            CacheOption::default().root(root).name("autotune"),
        )
    }

    fn insert(
        cache: &mut Cache<PersistentCacheKey<String>, PersistentCacheValue>,
        key: &str,
        checksum: &str,
        fastest_index: usize,
    ) {
        cache
            .insert(
                PersistentCacheKey {
                    key: key.to_owned(),
                    checksum: checksum.to_owned(),
                },
                PersistentCacheValue {
                    fastest_index,
                    results: Vec::new(),
                },
            )
            .expect("Cache insert should succeed");
    }

    #[test]
    fn test_tune_cache_seed_overlay() {
        let seed_root = temp_root("seed");
        let writable_root = temp_root("writable");
        let mut seed = persistent_cache(&seed_root);
        insert(&mut seed, "seed-only", "seed-checksum", 1);
        insert(&mut seed, "overlap", "seed-overlap-checksum", 2);
        let mut writable = persistent_cache(&writable_root);
        insert(&mut writable, "overlap", "writable-overlap-checksum", 3);

        let mut cache = TuneCache::<String>::new_with_roots(
            "test",
            "device",
            writable_root.clone(),
            Some(seed_root.clone()),
        );

        assert!(matches!(
            cache.validate_checksum(&"seed-only".to_owned(), "seed-checksum"),
            TuneCacheResult::Hit { fastest_index: 1 },
        ));
        assert!(matches!(
            cache.validate_checksum(&"overlap".to_owned(), "writable-overlap-checksum"),
            TuneCacheResult::Hit { fastest_index: 3 },
        ));

        cache.persistent_cache_insert(
            "new".to_owned(),
            "new-checksum".to_owned(),
            4,
            Vec::<AutotuneResult>::new(),
        );
        let seed = persistent_cache(&seed_root);
        let writable = persistent_cache(&writable_root);
        let new_key = PersistentCacheKey {
            key: "new".to_owned(),
            checksum: "new-checksum".to_owned(),
        };
        assert!(seed.get(&new_key).is_none());
        assert_eq!(
            writable.get(&new_key).map(|value| value.fastest_index),
            Some(4)
        );

        remove_dir_all(seed_root).ok();
        remove_dir_all(writable_root).ok();
    }
}
