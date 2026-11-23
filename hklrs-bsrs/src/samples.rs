//! Samples by name and the current one: `current` is a key of `entries`
//! after every operation, so there is always a sample to compute with.

use std::collections::BTreeMap;

use hklrs::{Sample, Wavelength};

use crate::DiffError;

/// A reflection as measured: Miller indices, every axis value in geometry
/// order (degrees for rotations, metres for translations) and the
/// wavelength it was measured at.
#[derive(Clone, Debug, PartialEq)]
pub struct Measured {
    pub hkl: [f64; 3],
    pub positions: Vec<f64>,
    pub wavelength: Wavelength,
}

#[derive(Clone, Debug)]
pub(crate) struct Entry {
    pub sample: Sample,
    pub reflections: BTreeMap<String, Measured>,
}

impl Entry {
    pub fn reflection(&self, name: &str) -> Result<&Measured, DiffError> {
        self.reflections
            .get(name)
            .ok_or_else(|| DiffError::UnknownReflection {
                sample: self.sample.name.clone(),
                name: name.to_owned(),
                known: self.reflections.keys().cloned().collect(),
            })
    }
}

#[derive(Clone, Debug)]
pub(crate) struct Samples {
    entries: BTreeMap<String, Entry>,
    current: String,
}

impl Samples {
    pub fn new(sample: Sample) -> Self {
        let current = sample.name.clone();
        let entry = Entry {
            sample,
            reflections: BTreeMap::new(),
        };
        Self {
            entries: BTreeMap::from([(current.clone(), entry)]),
            current,
        }
    }

    /// `entries` by sample name, `current` among them.
    pub fn from_entries(entries: Vec<Entry>, current: &str) -> Result<Self, DiffError> {
        let mut by_name = BTreeMap::new();
        for entry in entries {
            let name = entry.sample.name.clone();
            if by_name.insert(name.clone(), entry).is_some() {
                return Err(DiffError::DuplicateSample { name });
            }
        }
        if !by_name.contains_key(current) {
            return Err(DiffError::UnknownSample {
                name: current.to_owned(),
                known: by_name.into_keys().collect(),
            });
        }
        Ok(Self {
            entries: by_name,
            current: current.to_owned(),
        })
    }

    pub fn entries(&self) -> impl Iterator<Item = &Entry> {
        self.entries.values()
    }

    pub fn current_name(&self) -> &str {
        &self.current
    }

    pub fn names(&self) -> Vec<String> {
        self.entries.keys().cloned().collect()
    }

    pub fn current(&self) -> &Entry {
        &self.entries[&self.current]
    }

    pub fn current_mut(&mut self) -> &mut Entry {
        self.entries
            .get_mut(&self.current)
            .expect("current is always a key")
    }

    fn unknown(&self, name: &str) -> DiffError {
        DiffError::UnknownSample {
            name: name.to_owned(),
            known: self.names(),
        }
    }

    pub fn add(&mut self, sample: Sample) -> Result<(), DiffError> {
        if self.entries.contains_key(&sample.name) {
            return Err(DiffError::DuplicateSample { name: sample.name });
        }
        let entry = Entry {
            sample,
            reflections: BTreeMap::new(),
        };
        self.entries.insert(entry.sample.name.clone(), entry);
        Ok(())
    }

    pub fn select(&mut self, name: &str) -> Result<(), DiffError> {
        if !self.entries.contains_key(name) {
            return Err(self.unknown(name));
        }
        name.clone_into(&mut self.current);
        Ok(())
    }

    /// Removes `name`; the current sample changes only when it is the one
    /// removed (PY2-3), to the first remaining name.
    pub fn remove(&mut self, name: &str) -> Result<Sample, DiffError> {
        if !self.entries.contains_key(name) {
            return Err(self.unknown(name));
        }
        if self.entries.len() == 1 {
            return Err(DiffError::LastSample {
                name: name.to_owned(),
            });
        }
        let entry = self.entries.remove(name).expect("checked above");
        if self.current == name {
            self.current = self.entries.keys().next().expect("one left").clone();
        }
        Ok(entry.sample)
    }
}
