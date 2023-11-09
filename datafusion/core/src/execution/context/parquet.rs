// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

use std::sync::Arc;

use crate::datasource::physical_plan::parquet::plan_to_parquet;
use parquet::file::properties::WriterProperties;

use super::super::options::{ParquetReadOptions, ReadOptions};
use super::{DataFilePaths, DataFrame, ExecutionPlan, Result, SessionContext};

impl SessionContext {
    /// Creates a [`DataFrame`] for reading a Parquet data source.
    ///
    /// For more control such as reading multiple files, you can use
    /// [`read_table`](Self::read_table) with a [`super::ListingTable`].
    ///
    /// For an example, see [`read_csv`](Self::read_csv)
    pub async fn read_parquet<P: DataFilePaths>(
        &self,
        table_paths: P,
        options: ParquetReadOptions<'_>,
    ) -> Result<DataFrame> {
        self._read_type(table_paths, options).await
    }

    /// Registers a Parquet file as a table that can be referenced from SQL
    /// statements executed against this context.
    pub async fn register_parquet(
        &self,
        name: &str,
        table_path: &str,
        options: ParquetReadOptions<'_>,
    ) -> Result<()> {
        let listing_options = options.to_listing_options(&self.state.read().config);

        self.register_listing_table(
            name,
            table_path,
            listing_options,
            options.schema.map(|s| Arc::new(s.to_owned())),
            None,
        )
        .await?;
        Ok(())
    }

    /// Executes a query and writes the results to a partitioned Parquet file.
    pub async fn write_parquet(
        &self,
        plan: Arc<dyn ExecutionPlan>,
        path: impl AsRef<str>,
        writer_properties: Option<WriterProperties>,
    ) -> Result<()> {
        plan_to_parquet(self.task_ctx(), plan, path, writer_properties).await
    }
}

#[cfg(test)]
mod tests {
    use async_trait::async_trait;

    use crate::arrow::array::{Float32Array, Int32Array};
    use crate::arrow::datatypes::{DataType, Field, Schema};
    use crate::arrow::record_batch::RecordBatch;
    use crate::dataframe::DataFrameWriteOptions;
    use crate::parquet::basic::Compression;
    use crate::test_util::parquet_test_data;

    use super::*;

    #[tokio::test]
    async fn read_with_glob_path() -> Result<()> {
        let ctx = SessionContext::new();

        let df = ctx
            .read_parquet(
                format!("{}/alltypes_plain*.parquet", parquet_test_data()),
                ParquetReadOptions::default(),
            )
            .await?;
        let results = df.collect().await?;
        let total_rows: usize = results.iter().map(|rb| rb.num_rows()).sum();
        // alltypes_plain.parquet = 8 rows, alltypes_plain.snappy.parquet = 2 rows, alltypes_dictionary.parquet = 2 rows
        assert_eq!(total_rows, 10);
        Ok(())
    }

    #[tokio::test]
    async fn read_with_glob_path_issue_2465() -> Result<()> {
        let ctx = SessionContext::new();

        let df = ctx
            .read_parquet(
                // it was reported that when a path contains // (two consecutive separator) no files were found
                // in this test, regardless of parquet_test_data() value, our path now contains a //
                format!("{}/..//*/alltypes_plain*.parquet", parquet_test_data()),
                ParquetReadOptions::default(),
            )
            .await?;
        let results = df.collect().await?;
        let total_rows: usize = results.iter().map(|rb| rb.num_rows()).sum();
        // alltypes_plain.parquet = 8 rows, alltypes_plain.snappy.parquet = 2 rows, alltypes_dictionary.parquet = 2 rows
        assert_eq!(total_rows, 10);
        Ok(())
    }

    #[tokio::test]
    async fn read_from_registered_table_with_glob_path() -> Result<()> {
        let ctx = SessionContext::new();

        ctx.register_parquet(
            "test",
            &format!("{}/alltypes_plain*.parquet", parquet_test_data()),
            ParquetReadOptions::default(),
        )
        .await?;
        let df = ctx.sql("SELECT * FROM test").await?;
        let results = df.collect().await?;
        let total_rows: usize = results.iter().map(|rb| rb.num_rows()).sum();
        // alltypes_plain.parquet = 8 rows, alltypes_plain.snappy.parquet = 2 rows, alltypes_dictionary.parquet = 2 rows
        assert_eq!(total_rows, 10);
        Ok(())
    }

    #[tokio::test]
    async fn read_from_different_file_extension() -> Result<()> {
        let ctx = SessionContext::new();

        // Make up a new dataframe.
        let write_df = ctx.read_batch(RecordBatch::try_new(
            Arc::new(Schema::new(vec![
                Field::new("purchase_id", DataType::Int32, false),
                Field::new("price", DataType::Float32, false),
                Field::new("quantity", DataType::Int32, false),
            ])),
            vec![
                Arc::new(Int32Array::from(vec![1, 2, 3, 4, 5])),
                Arc::new(Float32Array::from(vec![1.12, 3.40, 2.33, 9.10, 6.66])),
                Arc::new(Int32Array::from(vec![1, 3, 2, 4, 3])),
            ],
        )?)?;

        // Write the dataframe to a parquet file named 'output1.parquet'
        write_df
            .clone()
            .write_parquet(
                "output1.parquet",
                DataFrameWriteOptions::new().with_single_file_output(true),
                Some(
                    WriterProperties::builder()
                        .set_compression(Compression::SNAPPY)
                        .build(),
                ),
            )
            .await?;

        // Write the dataframe to a parquet file named 'output2.parquet.snappy'
        write_df
            .clone()
            .write_parquet(
                "output2.parquet.snappy",
                DataFrameWriteOptions::new().with_single_file_output(true),
                Some(
                    WriterProperties::builder()
                        .set_compression(Compression::SNAPPY)
                        .build(),
                ),
            )
            .await?;

        // Write the dataframe to a parquet file named 'output3.parquet.snappy.parquet'
        write_df
            .write_parquet(
                "output3.parquet.snappy.parquet",
                DataFrameWriteOptions::new().with_single_file_output(true),
                Some(
                    WriterProperties::builder()
                        .set_compression(Compression::SNAPPY)
                        .build(),
                ),
            )
            .await?;

        // Read the dataframe from 'output1.parquet' with the default file extension.
        let read_df = ctx
            .read_parquet(
                "output1.parquet",
                ParquetReadOptions {
                    ..Default::default()
                },
            )
            .await?;

        let results = read_df.collect().await?;
        let total_rows: usize = results.iter().map(|rb| rb.num_rows()).sum();
        assert_eq!(total_rows, 5);

        // Read the dataframe from 'output2.parquet.snappy' with the correct file extension.
        let read_df = ctx
            .read_parquet(
                "output2.parquet.snappy",
                ParquetReadOptions {
                    file_extension: "snappy",
                    ..Default::default()
                },
            )
            .await?;
        let results = read_df.collect().await?;
        let total_rows: usize = results.iter().map(|rb| rb.num_rows()).sum();
        assert_eq!(total_rows, 5);

        // Read the dataframe from 'output3.parquet.snappy.parquet' with the wrong file extension.
        let read_df = ctx
            .read_parquet(
                "output2.parquet.snappy",
                ParquetReadOptions {
                    ..Default::default()
                },
            )
            .await;

        assert_eq!(
            read_df.unwrap_err().strip_backtrace(),
            "Execution error: File 'output2.parquet.snappy' does not match the expected extension '.parquet'"
        );

        // Read the dataframe from 'output3.parquet.snappy.parquet' with the correct file extension.
        let read_df = ctx
            .read_parquet(
                "output3.parquet.snappy.parquet",
                ParquetReadOptions {
                    ..Default::default()
                },
            )
            .await?;

        let results = read_df.collect().await?;
        let total_rows: usize = results.iter().map(|rb| rb.num_rows()).sum();
        assert_eq!(total_rows, 5);
        Ok(())
    }

    // Test for compilation error when calling read_* functions from an #[async_trait] function.
    // See https://github.com/apache/arrow-datafusion/issues/1154
    #[async_trait]
    trait CallReadTrait {
        async fn call_read_parquet(&self) -> DataFrame;
    }

    struct CallRead {}

    #[async_trait]
    impl CallReadTrait for CallRead {
        async fn call_read_parquet(&self) -> DataFrame {
            let ctx = SessionContext::new();
            ctx.read_parquet("dummy", ParquetReadOptions::default())
                .await
                .unwrap()
        }
    }
}
