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

//! Defines physical expressions that can evaluated at runtime during query execution

use crate::aggregate::groups_accumulator::accumulate::NullState;
use crate::aggregate::utils::down_cast_any_ref;
use crate::expressions::format_state_name;
use crate::{AggregateExpr, EmitTo, GroupsAccumulator, PhysicalExpr};
use arrow::array::ArrayRef;
use arrow::datatypes::{DataType, Field};
use arrow_array::builder::{ListBuilder, PrimitiveBuilder};
use arrow_array::cast::AsArray;
use arrow_array::types::{
    Float32Type, Float64Type, Int16Type, Int32Type, Int64Type, Int8Type, UInt16Type,
    UInt32Type, UInt64Type, UInt8Type,
};
use arrow_array::{Array, ArrowPrimitiveType, BooleanArray, ListArray, PrimitiveArray};
use datafusion_common::cast::as_list_array;
use datafusion_common::utils::array_into_list_array;
use datafusion_common::ScalarValue;
use datafusion_common::{DataFusionError, Result};
use datafusion_expr::Accumulator;
use std::any::Any;
use std::sync::Arc;

/// ARRAY_AGG aggregate expression
#[derive(Debug)]
pub struct ArrayAgg {
    /// Column name
    name: String,
    /// The DataType for the input expression
    input_data_type: DataType,
    /// The input expression
    expr: Arc<dyn PhysicalExpr>,
    /// If the input expression can have NULLs
    nullable: bool,
}

impl ArrayAgg {
    /// Create a new ArrayAgg aggregate function
    pub fn new(
        expr: Arc<dyn PhysicalExpr>,
        name: impl Into<String>,
        data_type: DataType,
        nullable: bool,
    ) -> Self {
        Self {
            name: name.into(),
            input_data_type: data_type,
            expr,
            nullable,
        }
    }
}

impl AggregateExpr for ArrayAgg {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn field(&self) -> Result<Field> {
        Ok(Field::new_list(
            &self.name,
            // This should be the same as return type of AggregateFunction::ArrayAgg
            Field::new("item", self.input_data_type.clone(), true),
            self.nullable,
        ))
    }

    fn create_accumulator(&self) -> Result<Box<dyn Accumulator>> {
        Ok(Box::new(ArrayAggAccumulator::try_new(
            &self.input_data_type,
        )?))
    }

    fn state_fields(&self) -> Result<Vec<Field>> {
        Ok(vec![Field::new_list(
            format_state_name(&self.name, "array_agg"),
            Field::new("item", self.input_data_type.clone(), true),
            self.nullable,
        )])
    }

    fn expressions(&self) -> Vec<Arc<dyn PhysicalExpr>> {
        vec![self.expr.clone()]
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn groups_accumulator_supported(&self) -> bool {
        self.input_data_type.is_primitive()
    }

    fn create_groups_accumulator(&self) -> Result<Box<dyn GroupsAccumulator>> {
        match self.input_data_type {
            DataType::Int8 => Ok(Box::new(ArrayAggGroupsAccumulator::<Int8Type>::new())),
            DataType::Int16 => {
                Ok(Box::new(ArrayAggGroupsAccumulator::<Int16Type>::new()))
            }
            DataType::Int32 => {
                Ok(Box::new(ArrayAggGroupsAccumulator::<Int32Type>::new()))
            }
            DataType::Int64 => {
                Ok(Box::new(ArrayAggGroupsAccumulator::<Int64Type>::new()))
            }
            DataType::UInt8 => {
                Ok(Box::new(ArrayAggGroupsAccumulator::<UInt8Type>::new()))
            }
            DataType::UInt16 => {
                Ok(Box::new(ArrayAggGroupsAccumulator::<UInt16Type>::new()))
            }
            DataType::UInt32 => {
                Ok(Box::new(ArrayAggGroupsAccumulator::<UInt32Type>::new()))
            }
            DataType::UInt64 => {
                Ok(Box::new(ArrayAggGroupsAccumulator::<UInt64Type>::new()))
            }
            DataType::Float32 => {
                Ok(Box::new(ArrayAggGroupsAccumulator::<Float32Type>::new()))
            }
            DataType::Float64 => {
                Ok(Box::new(ArrayAggGroupsAccumulator::<Float64Type>::new()))
            }
            _ => Err(DataFusionError::Internal(format!(
                "ArrayAggGroupsAccumulator not supported for data type {:?}",
                self.input_data_type
            ))),
        }
    }
}

impl PartialEq<dyn Any> for ArrayAgg {
    fn eq(&self, other: &dyn Any) -> bool {
        down_cast_any_ref(other)
            .downcast_ref::<Self>()
            .map(|x| {
                self.name == x.name
                    && self.input_data_type == x.input_data_type
                    && self.expr.eq(&x.expr)
            })
            .unwrap_or(false)
    }
}

#[derive(Debug)]
pub(crate) struct ArrayAggAccumulator {
    values: Vec<ArrayRef>,
    datatype: DataType,
}

impl ArrayAggAccumulator {
    /// new array_agg accumulator based on given item data type
    pub fn try_new(datatype: &DataType) -> Result<Self> {
        Ok(Self {
            values: vec![],
            datatype: datatype.clone(),
        })
    }
}

impl Accumulator for ArrayAggAccumulator {
    // Append value like Int64Array(1,2,3)
    fn update_batch(&mut self, values: &[ArrayRef]) -> Result<()> {
        if values.is_empty() {
            return Ok(());
        }
        assert!(values.len() == 1, "array_agg can only take 1 param!");
        let val = values[0].clone();
        self.values.push(val);
        Ok(())
    }

    // Append value like ListArray(Int64Array(1,2,3), Int64Array(4,5,6))
    fn merge_batch(&mut self, states: &[ArrayRef]) -> Result<()> {
        if states.is_empty() {
            return Ok(());
        }
        assert!(states.len() == 1, "array_agg states must be singleton!");

        let list_arr = as_list_array(&states[0])?;
        for arr in list_arr.iter().flatten() {
            self.values.push(arr);
        }
        Ok(())
    }

    fn state(&self) -> Result<Vec<ScalarValue>> {
        Ok(vec![self.evaluate()?])
    }

    fn evaluate(&self) -> Result<ScalarValue> {
        // Transform Vec<ListArr> to ListArr

        let element_arrays: Vec<&dyn Array> =
            self.values.iter().map(|a| a.as_ref()).collect();

        if element_arrays.is_empty() {
            let arr = ScalarValue::new_list(&[], &self.datatype);
            return Ok(ScalarValue::List(arr));
        }

        let concated_array = arrow::compute::concat(&element_arrays)?;
        let list_array = array_into_list_array(concated_array);

        Ok(ScalarValue::List(Arc::new(list_array)))
    }

    fn size(&self) -> usize {
        std::mem::size_of_val(self)
            + (std::mem::size_of::<ArrayRef>() * self.values.capacity())
            + self
                .values
                .iter()
                .map(|arr| arr.get_array_memory_size())
                .sum::<usize>()
            + self.datatype.size()
            - std::mem::size_of_val(&self.datatype)
    }
}

struct ArrayAggGroupsAccumulator<T>
where
    T: ArrowPrimitiveType + Send,
{
    values: Vec<Vec<Option<<T as ArrowPrimitiveType>::Native>>>,
    null_state: NullState,
}

impl<T> ArrayAggGroupsAccumulator<T>
where
    T: ArrowPrimitiveType + Send,
{
    pub fn new() -> Self {
        Self {
            values: vec![],
            null_state: NullState::new(),
        }
    }
}

impl<T> ArrayAggGroupsAccumulator<T>
where
    T: ArrowPrimitiveType + Send,
{
    fn build_list(&mut self, emit_to: EmitTo) -> Result<ArrayRef> {
        let array = emit_to.take_needed(&mut self.values);
        let nulls = self.null_state.build(emit_to);

        assert_eq!(array.len(), nulls.len());

        let mut builder =
            ListBuilder::with_capacity(PrimitiveBuilder::<T>::new(), nulls.len());
        for (is_valid, arr) in nulls.iter().zip(array.iter()) {
            if is_valid {
                for value in arr.iter() {
                    builder.values().append_option(*value);
                }
                builder.append(true);
            } else {
                builder.append_null();
            }
        }

        Ok(Arc::new(builder.finish()))
    }
}

impl<T> GroupsAccumulator for ArrayAggGroupsAccumulator<T>
where
    T: ArrowPrimitiveType + Send + Sync,
{
    // TODO:
    // 1. Implement support for null state
    // 2. Implement support for low level ListArray creation api with offsets and nulls
    // 3. Implement support for variable size types such as Utf8
    // 4. Implement support for accumulating Lists of any level of nesting
    // 5. Use this group accumulator in array_agg_distinct.rs

    fn update_batch(
        &mut self,
        values: &[ArrayRef],
        group_indices: &[usize],
        opt_filter: Option<&BooleanArray>,
        total_num_groups: usize,
    ) -> Result<()> {
        assert_eq!(values.len(), 1, "single argument to update_batch");
        let values = values[0].as_primitive::<T>();

        self.values.resize(total_num_groups, vec![]);

        self.null_state.accumulate(
            group_indices,
            values,
            opt_filter,
            total_num_groups,
            |group_index, new_value| {
                self.values[group_index].push(Some(new_value));
            },
        );

        Ok(())
    }

    fn merge_batch(
        &mut self,
        values: &[ArrayRef],
        group_indices: &[usize],
        opt_filter: Option<&BooleanArray>,
        total_num_groups: usize,
    ) -> Result<()> {
        assert_eq!(values.len(), 1, "single argument to merge_batch");
        let values = values[0].as_list();

        self.values.resize(total_num_groups, vec![]);

        self.null_state.accumulate_array(
            group_indices,
            values,
            opt_filter,
            total_num_groups,
            |group_index, new_value: &PrimitiveArray<T>| {
                self.values[group_index].append(
                    new_value
                        .into_iter()
                        .collect::<Vec<Option<T::Native>>>()
                        .as_mut(),
                );
            },
        );

        Ok(())
    }

    fn evaluate(&mut self, emit_to: EmitTo) -> Result<ArrayRef> {
        Ok(self.build_list(emit_to)?)
    }

    fn state(&mut self, emit_to: EmitTo) -> Result<Vec<ArrayRef>> {
        Ok(vec![self.build_list(emit_to)?])
    }

    fn size(&self) -> usize {
        self.values.capacity()
            + self.values.iter().map(|arr| arr.capacity()).sum::<usize>()
                * std::mem::size_of::<<T as ArrowPrimitiveType>::Native>()
            + self.null_state.size()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expressions::col;
    use crate::expressions::tests::{aggregate, aggregate_new};
    use arrow::array::ArrayRef;
    use arrow::array::Int32Array;
    use arrow::datatypes::*;
    use arrow::record_batch::RecordBatch;
    use arrow_array::Array;
    use arrow_array::ListArray;
    use arrow_buffer::OffsetBuffer;
    use datafusion_common::DataFusionError;
    use datafusion_common::Result;

    macro_rules! test_op {
        ($ARRAY:expr, $DATATYPE:expr, $OP:ident, $EXPECTED:expr) => {
            test_op!($ARRAY, $DATATYPE, $OP, $EXPECTED, $EXPECTED.data_type())
        };
        ($ARRAY:expr, $DATATYPE:expr, $OP:ident, $EXPECTED:expr, $EXPECTED_DATATYPE:expr) => {{
            let schema = Schema::new(vec![Field::new("a", $DATATYPE, true)]);

            let batch = RecordBatch::try_new(Arc::new(schema.clone()), vec![$ARRAY])?;

            let agg = Arc::new(<$OP>::new(
                col("a", &schema)?,
                "bla".to_string(),
                $EXPECTED_DATATYPE,
                true,
            ));
            let actual = aggregate(&batch, agg)?;
            let expected = ScalarValue::from($EXPECTED);

            assert_eq!(expected, actual);

            Ok(()) as Result<(), DataFusionError>
        }};
    }

    macro_rules! test_op_new {
        ($ARRAY:expr, $DATATYPE:expr, $OP:ident, $EXPECTED:expr) => {
            generic_test_op_new!(
                $ARRAY,
                $DATATYPE,
                $OP,
                $EXPECTED,
                $EXPECTED.data_type().clone()
            )
        };
        ($ARRAY:expr, $DATATYPE:expr, $OP:ident, $EXPECTED:expr, $EXPECTED_DATATYPE:expr) => {{
            let schema = Schema::new(vec![Field::new("a", $DATATYPE, true)]);

            let batch = RecordBatch::try_new(Arc::new(schema.clone()), vec![$ARRAY])?;

            let agg = Arc::new(<$OP>::new(
                col("a", &schema)?,
                "bla".to_string(),
                $EXPECTED_DATATYPE,
                true,
            ));
            let actual = aggregate_new(&batch, agg)?;
            assert_eq!($EXPECTED, &actual);

            Ok(()) as Result<(), DataFusionError>
        }};
    }

    #[test]
    fn array_agg_i32() -> Result<()> {
        let a: ArrayRef = Arc::new(Int32Array::from(vec![1, 2, 3, 4, 5]));

        let list = ListArray::from_iter_primitive::<Int32Type, _, _>(vec![Some(vec![
            Some(1),
            Some(2),
            Some(3),
            Some(4),
            Some(5),
        ])]);
        let expected = ScalarValue::List(Arc::new(list.clone()));

        test_op!(
            a.clone(),
            DataType::Int32,
            ArrayAgg,
            expected,
            DataType::Int32
        );

        let expected: ArrayRef = Arc::new(list);
        test_op_new!(a, DataType::Int32, ArrayAgg, &expected, DataType::Int32)
    }

    #[test]
    fn array_agg_nested() -> Result<()> {
        let a1 = ListArray::from_iter_primitive::<Int32Type, _, _>(vec![Some(vec![
            Some(1),
            Some(2),
            Some(3),
        ])]);
        let a2 = ListArray::from_iter_primitive::<Int32Type, _, _>(vec![Some(vec![
            Some(4),
            Some(5),
        ])]);
        let l1 = ListArray::new(
            Arc::new(Field::new("item", a1.data_type().to_owned(), true)),
            OffsetBuffer::from_lengths([a1.len() + a2.len()]),
            arrow::compute::concat(&[&a1, &a2])?,
            None,
        );

        let a1 =
            ListArray::from_iter_primitive::<Int32Type, _, _>(vec![Some(vec![Some(6)])]);
        let a2 = ListArray::from_iter_primitive::<Int32Type, _, _>(vec![Some(vec![
            Some(7),
            Some(8),
        ])]);
        let l2 = ListArray::new(
            Arc::new(Field::new("item", a1.data_type().to_owned(), true)),
            OffsetBuffer::from_lengths([a1.len() + a2.len()]),
            arrow::compute::concat(&[&a1, &a2])?,
            None,
        );

        let a1 =
            ListArray::from_iter_primitive::<Int32Type, _, _>(vec![Some(vec![Some(9)])]);
        let l3 = ListArray::new(
            Arc::new(Field::new("item", a1.data_type().to_owned(), true)),
            OffsetBuffer::from_lengths([a1.len()]),
            arrow::compute::concat(&[&a1])?,
            None,
        );

        let list = ListArray::new(
            Arc::new(Field::new("item", l1.data_type().to_owned(), true)),
            OffsetBuffer::from_lengths([l1.len() + l2.len() + l3.len()]),
            arrow::compute::concat(&[&l1, &l2, &l3])?,
            None,
        );
        let list = ScalarValue::List(Arc::new(list));
        let l1 = ScalarValue::List(Arc::new(l1));
        let l2 = ScalarValue::List(Arc::new(l2));
        let l3 = ScalarValue::List(Arc::new(l3));

        let array = ScalarValue::iter_to_array(vec![l1, l2, l3]).unwrap();

        test_op!(
            array,
            DataType::List(Arc::new(Field::new_list(
                "item",
                Field::new("item", DataType::Int32, true),
                true,
            ))),
            ArrayAgg,
            list,
            DataType::List(Arc::new(Field::new("item", DataType::Int32, true,)))
        )
    }
}
