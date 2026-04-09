# How to Add Your Real Metadata

## Step 1 — Copy your metadata.csv into the data/ folder

```
sheet_metal_project/
└── data/
    └── metadata.csv   ← put it here
```

## Step 2 — Run the pipeline normally

```bash
bash run_pipeline.sh
```

The script automatically detects metadata.csv and uses it.
No other changes needed.

---

## Expected Column Names

The script looks for these columns (case-insensitive):

| Column | Description | Example values |
|---|---|---|
| MAT | Material hardening factor | 0.9 – 1.1 |
| FC | Friction coefficient | 0.05 – 0.15 |
| SHTK | Sheet thickness (mm) | 0.95 – 1.00 |
| BF | Blank holder force (kN) | 100 – 500 |
| geometry | Geometry type (optional) | Concave/Convex/Rectangular |
| outcome | FLD outcome code (optional) | 0–4 |

## If your columns have different names

Edit the col_map section in 1_generate_data.py:

```python
col_map = {
    'your_mat_column':  'MAT',
    'your_fc_column':   'FC',
    'your_shtk_column': 'SHTK',
    'your_bf_column':   'BF',
}
```

---

## To Enable LLM in RAG (Optional)

Get a free Groq API key at https://console.groq.com

```bash
export GROQ_API_KEY="your_key_here"
bash run_pipeline.sh
```

Without an API key the RAG still works — it returns
the most relevant retrieved document directly.
