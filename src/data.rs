use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::fs;
use std::path::Path;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Split {
    Train,
    Validation,
}
pub type Batch = (Vec<Vec<usize>>, Vec<Vec<usize>>);

#[derive(Debug)]
pub enum DataError {
    Io(std::io::Error),
    TextTooShort { length: usize, block_size: usize },
    EmptyBatch,
}

impl fmt::Display for DataError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(error) => write!(f, "{error}"),
            Self::TextTooShort { length, block_size } => write!(
                f,
                "text length {length} is too short for block size {block_size} and a validation split"
            ),
            Self::EmptyBatch => write!(f, "batch size must be positive"),
        }
    }
}

impl std::error::Error for DataError {}

impl From<std::io::Error> for DataError {
    fn from(error: std::io::Error) -> Self {
        Self::Io(error)
    }
}

struct Random {
    state: u64,
}

impl Random {
    fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 { 0x9e3779b97f4a7c15 } else { seed },
        }
    }

    fn index(&mut self, limit: usize) -> usize {
        let mut value = self.state;
        value ^= value << 13;
        value ^= value >> 7;
        value ^= value << 17;
        self.state = value;
        value as usize % limit
    }
}

pub struct CharDataset {
    vocabulary: Vec<char>,
    train: Vec<usize>,
    validation: Vec<usize>,
    block_size: usize,
    train_random: Random,
    validation_random: Random,
}

impl CharDataset {
    pub fn from_file(
        path: impl AsRef<Path>,
        block_size: usize,
        seed: u64,
    ) -> Result<Self, DataError> {
        Self::from_text(&fs::read_to_string(path)?, block_size, seed)
    }

    pub fn from_text(text: &str, block_size: usize, seed: u64) -> Result<Self, DataError> {
        let characters: Vec<char> = text.chars().collect();
        let split = characters.len() * 9 / 10;
        if block_size == 0
            || split <= block_size
            || characters.len().saturating_sub(split) <= block_size
        {
            return Err(DataError::TextTooShort {
                length: characters.len(),
                block_size,
            });
        }
        let vocabulary: Vec<char> = characters
            .iter()
            .copied()
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect();
        let encoding: BTreeMap<char, usize> = vocabulary
            .iter()
            .copied()
            .enumerate()
            .map(|(index, character)| (character, index))
            .collect();
        let encoded: Vec<usize> = characters
            .iter()
            .map(|character| encoding[character])
            .collect();
        Ok(Self {
            vocabulary,
            train: encoded[..split].to_vec(),
            validation: encoded[split..].to_vec(),
            block_size,
            train_random: Random::new(seed),
            validation_random: Random::new(seed ^ 0xd1b54a32d192ed03),
        })
    }

    pub fn vocabulary(&self) -> &[char] {
        &self.vocabulary
    }

    pub fn vocab_size(&self) -> usize {
        self.vocabulary.len()
    }

    pub fn batch(&mut self, split: Split, batch_size: usize) -> Result<Batch, DataError> {
        if batch_size == 0 {
            return Err(DataError::EmptyBatch);
        }
        let (data, random) = match split {
            Split::Train => (&self.train, &mut self.train_random),
            Split::Validation => (&self.validation, &mut self.validation_random),
        };
        let window_count = data.len() - self.block_size;
        let mut inputs = Vec::with_capacity(batch_size);
        let mut targets = Vec::with_capacity(batch_size);
        for _ in 0..batch_size {
            let start = random.index(window_count);
            inputs.push(data[start..start + self.block_size].to_vec());
            targets.push(data[start + 1..start + self.block_size + 1].to_vec());
        }
        Ok((inputs, targets))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn text() -> String {
        "To be, or not to be.\n".repeat(40)
    }

    #[test]
    fn batches_are_reproducible_and_targets_are_offset() {
        let mut left = CharDataset::from_text(&text(), 8, 42).unwrap();
        let mut right = CharDataset::from_text(&text(), 8, 42).unwrap();
        let left_train = left.batch(Split::Train, 4).unwrap();
        let right_train = right.batch(Split::Train, 4).unwrap();
        assert_eq!(left_train, right_train);
        for (input, target) in left_train.0.iter().zip(&left_train.1) {
            assert_eq!(&input[1..], &target[..target.len() - 1]);
        }
        assert_eq!(
            left.batch(Split::Validation, 3).unwrap(),
            right.batch(Split::Validation, 3).unwrap()
        );
    }

    #[test]
    fn vocabulary_is_sorted_and_validation_is_held_out() {
        let mut dataset = CharDataset::from_text(&text(), 8, 7).unwrap();
        assert!(
            dataset
                .vocabulary()
                .windows(2)
                .all(|pair| pair[0] < pair[1])
        );
        assert!(dataset.vocab_size() > 1);
        let train = dataset.batch(Split::Train, 2).unwrap();
        let validation = dataset.batch(Split::Validation, 2).unwrap();
        assert_ne!(train, validation);
    }

    #[test]
    fn vendored_tinyshakespeare_has_expected_vocabulary() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("data/tinyshakespeare.txt");
        let mut dataset = CharDataset::from_file(path, 16, 123).unwrap();
        assert_eq!(dataset.vocab_size(), 65);
        let (inputs, targets) = dataset.batch(Split::Train, 8).unwrap();
        assert_eq!(inputs.len(), 8);
        assert!(inputs.iter().all(|sequence| sequence.len() == 16));
        assert!(targets.iter().all(|sequence| sequence.len() == 16));
    }
}
