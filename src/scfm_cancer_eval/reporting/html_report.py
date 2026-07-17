"""Generate a self-contained, interactive evaluation report."""

from __future__ import annotations

import json
from pathlib import Path

from scfm_cancer_eval.reporting.comparison import (
    _atomic_write_text,
    build_comparison_payload,
    build_comparison_records,
)
from scfm_cancer_eval.reporting.discovery import DiscoveryResult


def _embedded_json(payload: dict) -> str:
    return (
        json.dumps(payload, sort_keys=True, separators=(",", ":"))
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
    )


def render_html_report(
    discovery: DiscoveryResult,
    *,
    title: str = "scFM evaluation report",
) -> str:
    records = build_comparison_records(discovery)
    payload = build_comparison_payload(discovery, records)
    report_data = _embedded_json(payload)
    escaped_title = (
        title.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{escaped_title}</title>
  <style>
    :root {{
      color-scheme: light dark;
      --bg: #f7f7f5;
      --surface: #ffffff;
      --text: #20211f;
      --muted: #676a64;
      --line: #d9dbd5;
      --accent: #176b5b;
      --accent-soft: #dcece7;
      --danger: #9b3b32;
      --danger-soft: #f3e2df;
    }}
    @media (prefers-color-scheme: dark) {{
      :root {{
        --bg: #171816;
        --surface: #20221f;
        --text: #eceee9;
        --muted: #a9ada5;
        --line: #3a3d38;
        --accent: #72c7b4;
        --accent-soft: #263f39;
        --danger: #ef958b;
        --danger-soft: #482b28;
      }}
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font: 14px/1.5 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    main {{ width: min(1500px, 100%); margin: 0 auto; padding: 32px; }}
    header {{ display: flex; justify-content: space-between; gap: 24px; align-items: end; }}
    h1 {{ margin: 0; font-size: 24px; letter-spacing: -0.02em; }}
    h2 {{ margin: 0 0 12px; font-size: 17px; }}
    p {{ margin: 4px 0 0; color: var(--muted); }}
    a {{ color: var(--accent); }}
    .exports {{ display: flex; gap: 14px; white-space: nowrap; }}
    .summary {{
      display: grid;
      grid-template-columns: repeat(4, minmax(120px, 1fr));
      border: 1px solid var(--line);
      background: var(--surface);
      margin: 24px 0;
    }}
    .summary-item {{ padding: 16px 18px; border-right: 1px solid var(--line); }}
    .summary-item:last-child {{ border-right: 0; }}
    .summary-value {{ display: block; font-size: 24px; font-weight: 650; }}
    .summary-label {{ color: var(--muted); font-size: 12px; }}
    .controls {{
      display: grid;
      grid-template-columns: minmax(220px, 2fr) repeat(3, minmax(150px, 1fr));
      gap: 12px;
      margin-bottom: 16px;
    }}
    label {{ display: grid; gap: 5px; color: var(--muted); font-size: 12px; }}
    input, select {{
      width: 100%;
      padding: 9px 10px;
      border: 1px solid var(--line);
      border-radius: 4px;
      background: var(--surface);
      color: var(--text);
      font: inherit;
    }}
    .layout {{
      display: grid;
      grid-template-columns: minmax(0, 2fr) minmax(280px, 1fr);
      gap: 20px;
      align-items: start;
    }}
    .panel {{
      border: 1px solid var(--line);
      background: var(--surface);
      padding: 18px;
      min-width: 0;
    }}
    .table-wrap {{ overflow: auto; max-height: 65vh; }}
    table {{ width: 100%; border-collapse: collapse; white-space: nowrap; }}
    th, td {{
      padding: 9px 10px;
      border-bottom: 1px solid var(--line);
      text-align: left;
      vertical-align: top;
    }}
    th {{
      position: sticky;
      top: 0;
      background: var(--surface);
      color: var(--muted);
      font-size: 12px;
      cursor: pointer;
    }}
    td.numeric {{ font-variant-numeric: tabular-nums; text-align: right; }}
    .badge {{
      display: inline-block;
      border-radius: 999px;
      padding: 2px 8px;
      background: var(--accent-soft);
      color: var(--accent);
      font-size: 12px;
    }}
    .badge.failure {{ background: var(--danger-soft); color: var(--danger); }}
    .chart {{ display: grid; gap: 11px; margin-top: 14px; }}
    .bar-row {{ display: grid; grid-template-columns: minmax(90px, 1fr) 2fr auto; gap: 8px; align-items: center; }}
    .bar-label {{ overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
    .bar-track {{ height: 10px; background: var(--accent-soft); }}
    .bar-fill {{ height: 100%; background: var(--accent); min-width: 1px; }}
    .bar-value {{ min-width: 62px; text-align: right; font-variant-numeric: tabular-nums; }}
    .issues {{ margin-top: 20px; }}
    .issue {{ padding: 10px 0; border-bottom: 1px solid var(--line); }}
    .issue-path {{ color: var(--danger); font-family: ui-monospace, monospace; overflow-wrap: anywhere; }}
    .empty {{ color: var(--muted); padding: 20px 0; }}
    @media (max-width: 900px) {{
      main {{ padding: 20px; }}
      header {{ display: block; }}
      .exports {{ margin-top: 12px; }}
      .summary {{ grid-template-columns: repeat(2, 1fr); }}
      .summary-item:nth-child(2) {{ border-right: 0; }}
      .summary-item:nth-child(-n+2) {{ border-bottom: 1px solid var(--line); }}
      .controls, .layout {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
<main>
  <header>
    <div>
      <h1>{escaped_title}</h1>
      <p>Validated runs grouped by dataset, task, evaluation, and model.</p>
    </div>
    <nav class="exports" aria-label="Report exports">
      <a href="comparison.csv">Download CSV</a>
      <a href="comparison.json">Download JSON</a>
    </nav>
  </header>

  <section class="summary" aria-label="Report summary">
    <div class="summary-item"><span class="summary-value" id="run-count">0</span><span class="summary-label">Validated runs</span></div>
    <div class="summary-item"><span class="summary-value" id="model-count">0</span><span class="summary-label">Models</span></div>
    <div class="summary-item"><span class="summary-value" id="record-count">0</span><span class="summary-label">Evaluation records</span></div>
    <div class="summary-item"><span class="summary-value" id="issue-count">0</span><span class="summary-label">Discovery issues</span></div>
  </section>

  <section aria-labelledby="comparison-title">
    <h2 id="comparison-title">Comparison</h2>
    <div class="controls">
      <label>Search
        <input id="search" type="search" placeholder="Run, model, dataset, or task">
      </label>
      <label>Dataset
        <select id="dataset-filter"><option value="">All datasets</option></select>
      </label>
      <label>Evaluation
        <select id="kind-filter"><option value="">All evaluations</option></select>
      </label>
      <label>Chart metric
        <select id="metric-filter"></select>
      </label>
    </div>
    <div class="layout">
      <div class="panel">
        <div class="table-wrap">
          <table>
            <thead><tr id="table-head"></tr></thead>
            <tbody id="table-body"></tbody>
          </table>
          <div id="table-empty" class="empty" hidden>No records match the current filters.</div>
        </div>
      </div>
      <aside class="panel">
        <h2 id="chart-title">Metric comparison</h2>
        <p id="chart-caption">Select a numeric metric to compare matching records.</p>
        <div id="chart" class="chart"></div>
        <div id="chart-empty" class="empty">No numeric metrics are available.</div>
      </aside>
    </div>
  </section>

  <section id="issues-section" class="panel issues" hidden>
    <h2>Discovery issues</h2>
    <p>These files were skipped; valid runs remain in the comparison.</p>
    <div id="issues"></div>
  </section>
</main>

<script id="report-data" type="application/json">{report_data}</script>
<script>
(() => {{
  "use strict";
  const payload = JSON.parse(document.getElementById("report-data").textContent);
  const records = payload.records || [];
  const issues = payload.issues || [];
  const metricNames = [...new Set(records.flatMap(record => Object.keys(record.metrics || {{}})))].sort();
  const fixedColumns = [
    ["run_id", "Run"],
    ["model_id", "Model"],
    ["dataset_path", "Dataset"],
    ["task_id", "Task"],
    ["evaluation_kind", "Evaluation"],
    ["evaluation_variant", "Variant"],
    ["split", "Split"],
    ["evaluation_status", "Status"]
  ];
  const columns = fixedColumns.concat(metricNames.map(name => [`metric:${{name}}`, name]));
  let sortKey = "model_id";
  let sortDirection = 1;

  const byId = id => document.getElementById(id);
  byId("run-count").textContent = String(payload.summary.run_count);
  byId("model-count").textContent = String(new Set(records.map(record => record.model_id)).size);
  byId("record-count").textContent = String(payload.summary.record_count);
  byId("issue-count").textContent = String(payload.summary.issue_count);

  function addOptions(select, values) {{
    values.filter(Boolean).sort().forEach(value => {{
      const option = document.createElement("option");
      option.value = value;
      option.textContent = value;
      select.append(option);
    }});
  }}
  addOptions(byId("dataset-filter"), [...new Set(records.map(record => record.dataset_path))]);
  addOptions(byId("kind-filter"), [...new Set(records.map(record => record.evaluation_kind))]);
  if (metricNames.length) {{
    metricNames.forEach(name => {{
      const option = document.createElement("option");
      option.value = name;
      option.textContent = name;
      byId("metric-filter").append(option);
    }});
  }} else {{
    const option = document.createElement("option");
    option.value = "";
    option.textContent = "No numeric metrics";
    byId("metric-filter").append(option);
    byId("metric-filter").disabled = true;
  }}

  function valueFor(record, key) {{
    return key.startsWith("metric:") ? record.metrics?.[key.slice(7)] : record[key];
  }}

  function filteredRecords() {{
    const query = byId("search").value.trim().toLowerCase();
    const dataset = byId("dataset-filter").value;
    const kind = byId("kind-filter").value;
    return records.filter(record => {{
      const searchable = [
        record.run_id,
        record.model_id,
        record.dataset_path,
        record.task_id,
        record.evaluation_kind
      ].filter(Boolean).join(" ").toLowerCase();
      return (!query || searchable.includes(query))
        && (!dataset || record.dataset_path === dataset)
        && (!kind || record.evaluation_kind === kind);
    }}).sort((left, right) => {{
      const a = valueFor(left, sortKey);
      const b = valueFor(right, sortKey);
      if (a == null && b == null) return 0;
      if (a == null) return 1;
      if (b == null) return -1;
      if (typeof a === "number" && typeof b === "number") return (a - b) * sortDirection;
      return String(a).localeCompare(String(b)) * sortDirection;
    }});
  }}

  function renderHead() {{
    const head = byId("table-head");
    head.replaceChildren();
    columns.forEach(([key, label]) => {{
      const cell = document.createElement("th");
      cell.scope = "col";
      cell.textContent = label + (sortKey === key ? (sortDirection > 0 ? " ↑" : " ↓") : "");
      cell.addEventListener("click", () => {{
        if (sortKey === key) sortDirection *= -1;
        else {{ sortKey = key; sortDirection = 1; }}
        render();
      }});
      head.append(cell);
    }});
  }}

  function renderTable(rows) {{
    const body = byId("table-body");
    body.replaceChildren();
    rows.forEach(record => {{
      const row = document.createElement("tr");
      columns.forEach(([key]) => {{
        const cell = document.createElement("td");
        const value = valueFor(record, key);
        if (key === "evaluation_status") {{
          const badge = document.createElement("span");
          badge.className = "badge" + (String(value).toLowerCase() === "success" ? "" : " failure");
          badge.textContent = value ?? "";
          cell.append(badge);
        }} else {{
          cell.textContent = value == null
            ? ""
            : typeof value === "object"
              ? JSON.stringify(value)
              : String(value);
        }}
        if (key.startsWith("metric:") && typeof value === "number") cell.className = "numeric";
        row.append(cell);
      }});
      body.append(row);
    }});
    byId("table-empty").hidden = rows.length !== 0;
  }}

  function renderChart(rows) {{
    const chart = byId("chart");
    chart.replaceChildren();
    const metric = byId("metric-filter").value;
    const numeric = rows
      .map(record => [record, record.metrics?.[metric]])
      .filter(([, value]) => typeof value === "number" && Number.isFinite(value))
      .sort((left, right) => right[1] - left[1]);
    byId("chart-title").textContent = metric ? `${{metric}} by model` : "Metric comparison";
    byId("chart-caption").textContent = metric
      ? "Aggregate metric from each matching evaluation record."
      : "Select a numeric metric to compare matching records.";
    byId("chart-empty").hidden = numeric.length !== 0;
    if (!numeric.length) return;
    const values = numeric.map(([, value]) => value);
    const min = Math.min(0, ...values);
    const max = Math.max(...values);
    const span = max - min || 1;
    numeric.slice(0, 25).forEach(([record, value]) => {{
      const row = document.createElement("div");
      row.className = "bar-row";
      const label = document.createElement("span");
      label.className = "bar-label";
      label.title = `${{record.model_id}} · ${{record.run_id}}`;
      label.textContent = record.model_id;
      const track = document.createElement("div");
      track.className = "bar-track";
      const fill = document.createElement("div");
      fill.className = "bar-fill";
      fill.style.width = `${{Math.max(1, ((value - min) / span) * 100)}}%`;
      track.append(fill);
      const amount = document.createElement("span");
      amount.className = "bar-value";
      amount.textContent = Number(value).toLocaleString(undefined, {{ maximumSignificantDigits: 6 }});
      row.append(label, track, amount);
      chart.append(row);
    }});
  }}

  function renderIssues() {{
    if (!issues.length) return;
    byId("issues-section").hidden = false;
    const container = byId("issues");
    issues.forEach(issue => {{
      const item = document.createElement("div");
      item.className = "issue";
      const path = document.createElement("div");
      path.className = "issue-path";
      path.textContent = issue.path;
      const message = document.createElement("div");
      message.textContent = issue.message;
      item.append(path, message);
      container.append(item);
    }});
  }}

  function render() {{
    const rows = filteredRecords();
    renderHead();
    renderTable(rows);
    renderChart(rows);
  }}
  ["search", "dataset-filter", "kind-filter", "metric-filter"].forEach(id => {{
    byId(id).addEventListener(id === "search" ? "input" : "change", render);
  }});
  renderIssues();
  render();
}})();
</script>
</body>
</html>
"""


def write_html_report(
    discovery: DiscoveryResult,
    output_dir: str | Path,
    *,
    filename: str = "report.html",
    title: str = "scFM evaluation report",
) -> Path:
    report_path = Path(output_dir) / filename
    _atomic_write_text(report_path, render_html_report(discovery, title=title))
    return report_path
