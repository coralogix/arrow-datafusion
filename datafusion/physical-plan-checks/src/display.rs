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

//! Rendering plans as text in every way DataFusion displays them, with
//! panics and formatting errors caught, so that checks can report them and
//! compare the displays of two plans.

use std::fmt::{self, Write};
use std::panic::AssertUnwindSafe;

use datafusion_physical_plan::{DisplayFormatType, ExecutionPlan, displayable};

use crate::exec::panic_message;

/// A way DataFusion displays a plan
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DisplayKind {
    /// The node alone, formatted with `fmt_as` in this format, as each line
    /// of `EXPLAIN` shows it
    Node(DisplayFormatType),
    /// The node and its descendants as the tree renderer draws them
    /// (`displayable(plan).tree_render()`), as `EXPLAIN FORMAT TREE` shows them
    Tree,
}

impl DisplayKind {
    /// Every way to display a plan: the node alone in each
    /// [`DisplayFormatType`], and the tree renderer
    pub const ALL: [DisplayKind; 4] = [
        DisplayKind::Node(DisplayFormatType::Default),
        DisplayKind::Node(DisplayFormatType::Verbose),
        DisplayKind::Node(DisplayFormatType::TreeRender),
        DisplayKind::Tree,
    ];

    /// Whether the display includes the descendants of the node
    pub fn includes_children(&self) -> bool {
        matches!(self, DisplayKind::Tree)
    }
}

impl fmt::Display for DisplayKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DisplayKind::Node(format) => {
                write!(f, "fmt_as(DisplayFormatType::{format:?})")
            }
            DisplayKind::Tree => write!(f, "the tree renderer"),
        }
    }
}

/// Why a plan could not be displayed
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DisplayError {
    /// Formatting panicked, with this message
    Panicked(String),
    /// Formatting returned `fmt::Error` although writing to a `String` cannot
    /// fail, which makes `to_string` panic
    FormatError,
}

impl fmt::Display for DisplayError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DisplayError::Panicked(message) => write!(f, "panicked: {message}"),
            DisplayError::FormatError => write!(
                f,
                "returned fmt::Error, which makes to_string() panic, although writing \
                 to a String cannot fail"
            ),
        }
    }
}

/// `plan` displayed in the way `kind` says, or why it could not be
pub fn display(
    plan: &dyn ExecutionPlan,
    kind: DisplayKind,
) -> Result<String, DisplayError> {
    let mut text = String::new();
    let written = std::panic::catch_unwind(AssertUnwindSafe(|| match kind {
        DisplayKind::Node(format) => write!(text, "{}", FmtAs { plan, format }),
        DisplayKind::Tree => write!(text, "{}", displayable(plan).tree_render()),
    }));
    match written {
        Ok(Ok(())) => Ok(text),
        Ok(Err(fmt::Error)) => Err(DisplayError::FormatError),
        Err(panic) => Err(DisplayError::Panicked(panic_message(&panic))),
    }
}

/// Displays a node alone with `fmt_as`
struct FmtAs<'a> {
    plan: &'a dyn ExecutionPlan,
    format: DisplayFormatType,
}

impl fmt::Display for FmtAs<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.plan.fmt_as(self.format, f)
    }
}
