use memmap2::MmapOptions;
use numpy::{
    datetime::{units::Milliseconds, Datetime},
    ndarray::Array2,
    PyArray1, PyArray2,
};
use pyo3::{exceptions::PyValueError, prelude::*};
use std::fs::File;

use mimalloc::MiMalloc;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

fn days_before_year(year: i32) -> i32 {
    fn leaps_before(y: i32) -> i32 {
        let n = y - 1;
        n / 4 - n / 100 + n / 400
    }
    let base_year = 1970;
    let years = year - base_year;
    let leap_diff = leaps_before(year) - leaps_before(base_year);
    years * 365 + leap_diff
}

#[inline]
fn next_token<'a>(line: &'a [u8], idx: &mut usize) -> Option<&'a [u8]> {
    while *idx < line.len() && line[*idx].is_ascii_whitespace() {
        *idx += 1;
    }
    if *idx >= line.len() {
        return None;
    }
    let start = *idx;
    while *idx < line.len() && !line[*idx].is_ascii_whitespace() {
        *idx += 1;
    }
    Some(&line[start..*idx])
}

#[inline]
fn parse_i32_required(
    token: Option<&[u8]>,
    line_no: usize,
    path: &str,
    field: &'static str,
) -> PyResult<i32> {
    let t = token.ok_or_else(|| {
        PyValueError::new_err(format!(
            "Invalid file {} at line {}: missing {}",
            path, line_no, field
        ))
    })?;
    lexical_core::parse::<i32>(t).map_err(|_| {
        PyValueError::new_err(format!(
            "Invalid file {} at line {}: bad {}",
            path, line_no, field
        ))
    })
}

#[inline]
fn parse_f32_required(token: Option<&[u8]>, line_no: usize, path: &str) -> PyResult<f32> {
    let t = token.ok_or_else(|| {
        PyValueError::new_err(format!(
            "Invalid file {} at line {}: missing data field",
            path, line_no
        ))
    })?;
    lexical_core::parse::<f32>(t).map_err(|_| {
        PyValueError::new_err(format!(
            "Invalid file {} at line {}: bad data field",
            path, line_no
        ))
    })
}

#[inline]
fn next_line_start(line_end: usize, total: usize) -> usize {
    if line_end < total {
        line_end + 1
    } else {
        line_end
    }
}

#[pyfunction]
fn read_sts<'py>(
    py: Python<'py>,
    path: String,
) -> PyResult<(
    Bound<'py, PyArray1<Datetime<Milliseconds>>>,
    Bound<'py, PyArray2<f32>>,
)> {
    let file_stream = File::open(&path)?;
    let estimated_rows = file_stream
        .metadata()
        .map(|m| (m.len() / 96) as usize)
        .unwrap_or(2_800_000);

    let mmap = unsafe { MmapOptions::new().map(&file_stream)? };
    let bytes = &mmap[..];

    let (mut obj, mut end_obj) = (0usize, 0usize);

    let mut times: Vec<Datetime<Milliseconds>> = Vec::with_capacity(estimated_rows);
    let mut values: Vec<f32> = Vec::with_capacity(estimated_rows * 11);

    let mut line_start = 0usize;
    let mut line_no = 0usize;
    let mut in_header = true;

    while line_start < bytes.len() {
        let mut line_end = line_start;
        while line_end < bytes.len() && bytes[line_end] != b'\n' {
            line_end += 1;
        }

        let mut line = &bytes[line_start..line_end];
        if line.ends_with(b"\r") {
            line = &line[..line.len() - 1];
        }
        line_no += 1;

        if line.is_empty() {
            line_start = next_line_start(line_end, bytes.len());
            continue;
        }

        if in_header {
            if line.windows(10).any(|w| w == b"END_OBJECT") {
                end_obj += 1;
            } else if line.windows(6).any(|w| w == b"OBJECT") {
                obj += 1;
            }
            if obj > 0 && obj == end_obj {
                in_header = false;
            }
            line_start = next_line_start(line_end, bytes.len());
            continue;
        }

        let mut idx = 0usize;
        let year = parse_i32_required(next_token(line, &mut idx), line_no, &path, "year")?;
        let doy = parse_i32_required(next_token(line, &mut idx), line_no, &path, "doy")?;
        let hour = parse_i32_required(next_token(line, &mut idx), line_no, &path, "hour")?;
        let minute = parse_i32_required(next_token(line, &mut idx), line_no, &path, "minute")?;
        let second = parse_i32_required(next_token(line, &mut idx), line_no, &path, "second")?;
        let millisecond =
            parse_i32_required(next_token(line, &mut idx), line_no, &path, "millisecond")?;

        if next_token(line, &mut idx).is_none() {
            return Err(PyValueError::new_err(format!(
                "Invalid file {} at line {}: missing field 7",
                path, line_no
            )));
        }

        for _ in 0..11 {
            values.push(parse_f32_required(
                next_token(line, &mut idx),
                line_no,
                &path,
            )?);
        }

        if next_token(line, &mut idx).is_some() {
            return Err(PyValueError::new_err(format!(
                "Invalid file {} at line {}: too many fields",
                path, line_no
            )));
        }

        let days_from_epoch = days_before_year(year) + (doy - 1);
        let epoch_ms = days_from_epoch as i64 * 86_400_000
            + hour as i64 * 3_600_000
            + minute as i64 * 60_000
            + second as i64 * 1_000
            + millisecond as i64;
        times.push(Datetime::<Milliseconds>::from(epoch_ms));

        line_start = next_line_start(line_end, bytes.len());
    }

    if in_header {
        return Err(PyValueError::new_err(format!(
            "Invalid file {}: header END_OBJECT not found",
            path
        )));
    }

    if times.is_empty() {
        return Err(PyValueError::new_err(format!(
            "Invalid file {}: no data rows found",
            path
        )));
    }

    let n_rows = times.len();
    let data = Array2::from_shape_vec((n_rows, 11), values)
        .map_err(|e| PyValueError::new_err(format!("Invalid data shape: {e}")))?;

    let arr_t = PyArray1::from_vec(py, times);
    let arr_d = PyArray2::from_owned_array(py, data);
    Ok((arr_t, arr_d))
}

#[pymodule]
fn sts_reader<'py>(_py: Python<'py>, m: &Bound<'py, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(read_sts, m)?)?;
    Ok(())
}
