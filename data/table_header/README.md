# Table-header row labels for RealKIE FCC Invoices

- `train_header_rows.json` / `test_header_rows.json`: per document id (see `*_meta.json`), the indices of the rows that form the line-item table header on the document's first line-item page. Rows are words grouped by y centre (`rows_idx()` in `scripts/build_table_header_dataset.py`).
- `*_meta.json`: document file name, page, row index in the HF dataset, and the first line-item row.
- `test_gold30_header_rows.json`: header rows for the first 30 test documents, labelled separately and used to check the labels and the Table Transformer baseline.

The labels were made from page images with an AI model (Claude) and checked against the 30-page set, where all 36 header rows matched. Same licence as RealKIE (CC BY-NC 4.0).
