# Part 1: Data Preparation

Extract and clean raw Scopus JSON data into structured format.

## Input
- 20,216 raw JSON files from Scopus API
- Fields: title, abstract, authors, citations, affiliations, etc.

## Processing
1. **Load JSON** - Parse all files
2. **Extract** - Pull relevant fields
3. **Clean** - Remove incomplete/invalid records
4. **Deduplicate** - Remove duplicate papers by title similarity
5. **Structure** - Organize into DataFrame

## Output
- `data/processed/papers.parquet` - 19,523 papers (29 MB)
- Fields: id, title, abstract, year, citations, authors, subjects

## Usage
```bash
jupyter notebook phase1_data_exploration_and_preparation.ipynb
```

## Key Stats
- Raw files: 20,216
- Final papers: 19,523
- Deduplication: ~700 duplicates removed
- Completeness: All papers have title + abstract
