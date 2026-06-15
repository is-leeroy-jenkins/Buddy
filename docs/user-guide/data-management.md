# Data Management

Data Management mode supports local SQLite-backed inspection, profiling, filtering, indexing,
visualization, and table operations.

## 🧭 Purpose

Data Management gives analysts a controlled interface for working with local tabular data. It
supports table discovery, schema review, row browsing, advanced filtering, aggregation,
visualization, indexing, import, export, and table maintenance workflows.

## 🧱 Workflow Position

Data Management is the local data layer of Buddy. It is useful before prompting a model, after
receiving analytical output, or when reviewing stored datasets that support financial management
analysis.

| Capability | Description |
| --- | --- |
| Table listing | Shows available SQLite tables. |
| Schema inspection | Displays columns, data types, and table structure. |
| Data browsing | Reads rows from selected tables. |
| Filtering | Applies column/operator/value filters. |
| Aggregation | Calculates counts, sums, averages, extrema, and medians. |
| Visualization | Generates charts for numeric and categorical data. |
| Profiling | Summarizes nulls, distinct values, and numeric ranges. |
| Maintenance | Supports table, column, index, and import/export operations. |

## 🧪 Example

```text
Task:
Inspect a budget execution table before asking the model to summarize anomalies.

Recommended setup:
- Mode: Data Management
- Select table: Authority or Outlays data
- Review schema
- Apply filters for fiscal year, account, or object class
- Generate profile and visualization
- Export or summarize results for downstream analysis
```

## ✅ Recommended Sequence

1. Open Data Management mode.
2. Select a table.
3. Review schema and row count.
4. Browse a limited sample before running operations.
5. Apply filters when the table is large.
6. Create an aggregation or visualization.
7. Profile the table for missing values and distinct values.
8. Export only the data needed for the next workflow.

## ⚠️ Safety Checks

Before modifying tables:

- Confirm the selected table name.
- Confirm the target column exists.
- Avoid destructive operations unless a backup exists.
- Validate custom SQL as read-only when using query tools.
- Use safe identifiers for table and column names.

## 🔗 Related API Modules

Use the `app` API documentation for table creation, schema inspection, filtering, visualization,
indexing, profile generation, and SQLite utility functions.
