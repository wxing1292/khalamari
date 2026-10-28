use std::collections::HashMap;
use std::path::Path;
use std::path::PathBuf;

use safetensors::SafeTensors;

use crate::checkpoint::index::SafeTensorIndex;
use crate::checkpoint::mapped_file::MappedFile;
use crate::checkpoint::tensor::TensorBytes;
use crate::checkpoint::tensor::TensorMap;
use crate::def::ModelExecutorError;

pub struct SafeTensorStore {
    model_dir: PathBuf,
    index: SafeTensorIndex,
    mapped_files: HashMap<PathBuf, MappedFile>,
}

impl SafeTensorStore {
    pub fn new(model_dir: impl AsRef<Path>, index: SafeTensorIndex) -> Self {
        Self {
            model_dir: model_dir.as_ref().to_path_buf(),
            index,
            mapped_files: HashMap::new(),
        }
    }

    pub fn from_model_dir(model_dir: impl AsRef<Path>) -> Result<Self, ModelExecutorError> {
        let index = SafeTensorIndex::load(&model_dir)?;
        Ok(Self::new(model_dir, index))
    }

    pub fn load(&mut self, file_name: &str) -> Result<(), ModelExecutorError> {
        let file_path = self.file_path(file_name);
        if let std::collections::hash_map::Entry::Vacant(entry) = self.mapped_files.entry(file_path) {
            let mapped = MappedFile::open(entry.key())?;
            entry.insert(mapped);
        }
        Ok(())
    }

    pub fn unload(&mut self, file_name: &str) {
        let file_path = self.file_path(file_name);
        self.mapped_files.remove(&file_path);
    }

    pub fn load_all(&mut self) -> Result<(), ModelExecutorError> {
        for file_name in self.index.file_names().map(ToOwned::to_owned).collect::<Vec<_>>() {
            self.load(&file_name)?;
        }
        Ok(())
    }

    pub fn unload_all(&mut self) {
        self.mapped_files.clear();
    }

    pub fn index(&self) -> &SafeTensorIndex {
        &self.index
    }

    pub fn tensor_bytes(
        &mut self,
        tensor_name: &str,
        dtype: safetensors::Dtype,
    ) -> Result<TensorBytes, ModelExecutorError> {
        self.read_tensor(tensor_name)?.expect_dtype(dtype)
    }

    pub fn load_tensors<'a>(
        &mut self,
        tensor_names: impl IntoIterator<Item = &'a str>,
    ) -> Result<TensorMap, ModelExecutorError> {
        let result = tensor_names
            .into_iter()
            .try_fold(TensorMap::new(), |mut tensors, tensor_name| {
                let tensor = self.read_tensor(tensor_name)?;
                if tensors.insert(tensor_name.to_string(), tensor).is_some() {
                    return Err(ModelExecutorError::custom(format!(
                        "duplicate tensor {tensor_name:?} in load request"
                    )));
                }
                Ok(tensors)
            });
        self.unload_all();
        result
    }

    fn read_tensor(&mut self, tensor_name: &str) -> Result<TensorBytes, ModelExecutorError> {
        let file_name = self.index().file_name_for(tensor_name)?.to_string();
        self.load(&file_name)?;
        let file_path = self.file_path(&file_name);
        let mapped = self
            .mapped_files
            .get(&file_path)
            .expect("safetensors file must be inserted before reading");
        let tensors = SafeTensors::deserialize(mapped.as_bytes()).map_err(|err| {
            ModelExecutorError::custom(format!(
                "unable to deserialize safetensors file {:?}, err: {err:?}",
                file_path
            ))
        })?;
        let view = tensors.tensor(tensor_name).map_err(|err| {
            ModelExecutorError::custom(format!(
                "unable to read tensor {tensor_name:?} from safetensors file {:?}, err: {err:?}",
                file_path
            ))
        })?;
        Ok(TensorBytes::from_view(tensor_name, &view))
    }

    fn file_path(&self, file_name: &str) -> PathBuf {
        self.model_dir.join(file_name)
    }
}

#[cfg(test)]
mod tests {
    use std::borrow::Cow;
    use std::collections::HashMap;
    use std::time::SystemTime;
    use std::time::UNIX_EPOCH;

    use safetensors::Dtype;
    use safetensors::tensor::View;
    use safetensors::tensor::serialize_to_file;

    use super::*;

    struct OwnedTensor {
        dtype: Dtype,
        shape: Vec<usize>,
        data: Vec<u8>,
    }

    impl View for &OwnedTensor {
        fn dtype(&self) -> Dtype {
            self.dtype
        }

        fn shape(&self) -> &[usize] {
            &self.shape
        }

        fn data(&self) -> Cow<'_, [u8]> {
            Cow::Borrowed(&self.data)
        }

        fn data_len(&self) -> usize {
            self.data.len()
        }
    }

    #[test]
    fn test_load_tensors_builds_typed_map() {
        let model_dir = temp_model_dir();
        let tensors = HashMap::from([
            (
                "a.weight".to_string(),
                OwnedTensor {
                    dtype: Dtype::U32,
                    shape: vec![2],
                    data: vec![0; 8],
                },
            ),
            (
                "b.weight".to_string(),
                OwnedTensor {
                    dtype: Dtype::BF16,
                    shape: vec![3],
                    data: vec![0; 6],
                },
            ),
        ]);
        serialize_to_file(
            tensors.iter().map(|(name, tensor)| (name.as_str(), tensor)),
            None,
            &model_dir.join("model.safetensors"),
        )
        .unwrap();
        std::fs::write(
            model_dir.join("model.safetensors.index.json"),
            serde_json::to_vec(&serde_json::json!({
                "weight_map": {
                    "a.weight": "model.safetensors",
                    "b.weight": "model.safetensors"
                }
            }))
            .unwrap(),
        )
        .unwrap();

        let mut store = SafeTensorStore::from_model_dir(&model_dir).unwrap();
        let tensors = store.load_tensors(["b.weight", "a.weight"]).unwrap();

        assert_eq!(tensors.get("a.weight").unwrap().dtype(), Dtype::U32);
        assert_eq!(tensors.get("a.weight").unwrap().shape(), [2]);
        assert_eq!(tensors.get("b.weight").unwrap().dtype(), Dtype::BF16);
        assert_eq!(tensors.get("b.weight").unwrap().shape(), [3]);
        assert!(store.mapped_files.is_empty());

        std::fs::remove_dir_all(model_dir).unwrap();
    }

    fn temp_model_dir() -> PathBuf {
        let path = std::env::temp_dir().join(format!(
            "khalamari-safetensor-store-test-{}-{}",
            std::process::id(),
            SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos()
        ));
        std::fs::create_dir_all(&path).unwrap();
        path
    }
}
