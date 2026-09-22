use super::czsc_signals::parse_signals_config;
use czsc_core::objects::bar::RawBar;
use czsc_core::objects::freq::Freq;
use czsc_core::objects::market::Market;
use czsc_trader::czsc_signals::CzscSignals;
use czsc_trader::sig_parse::get_signals_freqs;
use czsc_utils::bar_generator::BarGenerator;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use std::collections::HashMap;
use std::str::FromStr;

/// compact / edges_only 时跳过的行情/元数据列。
const COMPACT_SKIP_KEYS: &[&str] = &[
    "open", "high", "low", "close", "vol", "volume", "amount", "symbol", "id", "freq",
];

fn keep_key(key: &str, compact: bool) -> bool {
    if !compact {
        return true;
    }
    key == "dt" || !COMPACT_SKIP_KEYS.contains(&key)
}

fn is_signal_key(key: &str) -> bool {
    key != "dt" && !COMPACT_SKIP_KEYS.contains(&key)
}

/// 批量生成 CZSC 信号
///
/// 参数:
///   bars: 基础周期 K 线列表
///   signals_config: 信号配置列表
///   sdt: 信号开始计算日期，格式 "YYYYMMDD" 或 "YYYY-MM-DD"
///   init_n: 预热 K 线数量
///   df: 是否返回 DataFrame（默认 False 返回 list[dict]）
///   compact: True 时不导出 OHLCV/symbol/id/freq；内部列式组装
///   edges_only: True 时只导出信号状态边沿（dt/key/value），大幅减少导出量
#[pyfunction]
#[pyo3(signature = (
    bars,
    signals_config,
    sdt="20170101",
    init_n=500,
    df=false,
    compact=false,
    edges_only=false
))]
pub fn generate_czsc_signals(
    py: Python,
    bars: Vec<RawBar>,
    signals_config: &Bound<PyList>,
    sdt: &str,
    init_n: usize,
    df: bool,
    compact: bool,
    edges_only: bool,
) -> PyResult<Py<PyAny>> {
    if bars.is_empty() {
        return Err(PyValueError::new_err("bars 不能为空"));
    }

    // edges_only 隐含 compact（只跟踪信号列）
    let compact = compact || edges_only;

    let configs = parse_signals_config(signals_config)?;

    let freq_strs = get_signals_freqs(&configs);
    let freqs: Vec<Freq> = freq_strs
        .iter()
        .filter_map(|s| Freq::from_str(s).ok())
        .collect();

    let base_freq = bars[0].freq;
    let max_count = bars.len().max(1000);
    let bg = BarGenerator::new(base_freq, freqs, max_count, Market::Default)
        .map_err(|e| PyValueError::new_err(format!("创建 BarGenerator 失败: {e}")))?;

    let sdt_normalized = normalize_sdt(sdt);
    let mut split_idx = init_n.min(bars.len());

    for (i, bar) in bars.iter().enumerate() {
        let bar_date = bar.dt.format("%Y%m%d").to_string();
        if bar_date >= sdt_normalized && i >= init_n {
            split_idx = i;
            break;
        }
    }
    if split_idx >= bars.len() {
        split_idx = bars.len().saturating_sub(1);
    }

    let (bars_left, bars_right) = bars.split_at(split_idx);

    let symbol = bars[0].symbol.to_string();
    let mut signals = CzscSignals::new(symbol, bg);

    for bar in bars_left {
        signals.bg.update_bar(bar).map_err(|e| {
            PyValueError::new_err(format!("warmup 阶段 update_bar 失败 (dt={}): {e}", bar.dt))
        })?;
    }

    if let Some(last_warmup) = bars_left.last() {
        signals.prime_signals(last_warmup, &configs);
    }

    if edges_only {
        return collect_edges(py, &mut signals, &configs, bars_right, df);
    }

    collect_columns(py, &mut signals, &configs, bars_right, compact, df)
}

fn collect_edges(
    py: Python,
    signals: &mut CzscSignals,
    configs: &[czsc_trader::sig_parse::SignalConfig],
    bars_right: &[RawBar],
    df: bool,
) -> PyResult<Py<PyAny>> {
    let mut prev: HashMap<String, String> = HashMap::new();
    let mut dts: Vec<String> = Vec::new();
    let mut keys: Vec<String> = Vec::new();
    let mut values: Vec<String> = Vec::new();

    // prime 后的非「其他」状态作为边沿基线（不导出）
    for (k, v) in &signals.s {
        if is_signal_key(k) && !v.is_empty() && !v.contains("其他") {
            prev.insert(k.clone(), v.clone());
        }
    }

    for bar in bars_right {
        signals.update_signals(bar, configs).map_err(|e| {
            PyValueError::new_err(format!("update_signals 失败 (dt={}): {e}", bar.dt))
        })?;
        let dt = signals
            .s
            .get("dt")
            .cloned()
            .unwrap_or_else(|| bar.dt.to_rfc3339());

        for (k, v) in &signals.s {
            // 对齐 Python 边沿：含「其他」跳过且不刷新 prev
            if !is_signal_key(k) || v.is_empty() || v.contains("其他") {
                continue;
            }
            if prev.get(k).is_some_and(|old| old == v) {
                continue;
            }
            prev.insert(k.clone(), v.clone());
            dts.push(dt.clone());
            keys.push(k.clone());
            values.push(v.clone());
        }
    }

    let col_dict = PyDict::new(py);
    col_dict.set_item("dt", PyList::new(py, &dts)?)?;
    col_dict.set_item("key", PyList::new(py, &keys)?)?;
    col_dict.set_item("value", PyList::new(py, &values)?)?;

    if df {
        let pandas = py.import("pandas")?;
        let df_obj = pandas.call_method1("DataFrame", (col_dict,))?;
        Ok(df_obj.into_any().unbind())
    } else {
        let n = dts.len();
        let mut records: Vec<Py<PyAny>> = Vec::with_capacity(n);
        for i in 0..n {
            let row = PyDict::new(py);
            row.set_item("dt", &dts[i])?;
            row.set_item("key", &keys[i])?;
            row.set_item("value", &values[i])?;
            records.push(row.into_any().unbind());
        }
        let list = PyList::new(py, &records)?;
        Ok(list.into_any().unbind())
    }
}

fn collect_columns(
    py: Python,
    signals: &mut CzscSignals,
    configs: &[czsc_trader::sig_parse::SignalConfig],
    bars_right: &[RawBar],
    compact: bool,
    df: bool,
) -> PyResult<Py<PyAny>> {
    let n_right = bars_right.len();
    let mut col_order: Vec<String> = Vec::new();
    let mut columns: HashMap<String, Vec<String>> = HashMap::new();

    for bar in bars_right {
        signals.update_signals(bar, configs).map_err(|e| {
            PyValueError::new_err(format!("update_signals 失败 (dt={}): {e}", bar.dt))
        })?;

        if col_order.is_empty() {
            for (k, v) in &signals.s {
                if !keep_key(k, compact) {
                    continue;
                }
                col_order.push(k.clone());
                let mut series = Vec::with_capacity(n_right);
                series.push(v.clone());
                columns.insert(k.clone(), series);
            }
            continue;
        }

        for k in &col_order {
            let v = signals.s.get(k).cloned().unwrap_or_default();
            if let Some(series) = columns.get_mut(k) {
                series.push(v);
            }
        }
        for (k, v) in &signals.s {
            if !keep_key(k, compact) || columns.contains_key(k) {
                continue;
            }
            let rows_done = col_order
                .first()
                .and_then(|fk| columns.get(fk))
                .map(|s| s.len())
                .unwrap_or(0);
            let mut series = vec![String::new(); rows_done.saturating_sub(1)];
            series.push(v.clone());
            col_order.push(k.clone());
            columns.insert(k.clone(), series);
        }
    }

    let col_dict = PyDict::new(py);
    for k in &col_order {
        if let Some(vals) = columns.get(k) {
            let list = PyList::new(py, vals)?;
            col_dict.set_item(k, list)?;
        }
    }

    if df {
        let pandas = py.import("pandas")?;
        let df_obj = pandas.call_method1("DataFrame", (col_dict,))?;
        Ok(df_obj.into_any().unbind())
    } else {
        let n = columns.values().map(|s| s.len()).max().unwrap_or(0);
        let mut records: Vec<Py<PyAny>> = Vec::with_capacity(n);
        for i in 0..n {
            let row = PyDict::new(py);
            for k in &col_order {
                if let Some(vals) = columns.get(k) {
                    let v = vals.get(i).map(|s| s.as_str()).unwrap_or("");
                    row.set_item(k, v)?;
                }
            }
            records.push(row.into_any().unbind());
        }
        let list = PyList::new(py, &records)?;
        Ok(list.into_any().unbind())
    }
}

fn normalize_sdt(sdt: &str) -> String {
    sdt.replace('-', "")
}
