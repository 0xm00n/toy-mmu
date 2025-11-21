import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc
from datasets import load_from_disk
import argparse
import sys
from pathlib import Path


def load_table(file_path):
    """Load a PyArrow table from either a parquet file or datasets directory."""
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File or directory not found: {file_path}")

    # Try loading as parquet file
    if path.is_file() and path.suffix == ".parquet":
        return pq.read_table(file_path)

    # Try loading as datasets directory
    if path.is_dir():
        try:
            dataset = load_from_disk(file_path)
            return dataset.data.table  # Get the underlying Arrow table
        except Exception as e:
            raise ValueError(f"Could not load {file_path} as datasets directory: {e}")

    raise ValueError(f"Unsupported file type or format: {file_path}")


def compare_tables(table1, table2, label1="Table 1", label2="Table 2"):
    """Compare two PyArrow tables and report all differences."""
    issues = []

    print(f"\n{'='*70}")
    print(f"COMPARISON SUMMARY")
    print(f"{'='*70}")
    print(f"{label1}: {table1.num_rows} rows, {table1.num_columns} columns")
    print(f"{label2}: {table2.num_rows} rows, {table2.num_columns} columns")

    # Check row counts
    if table1.num_rows != table2.num_rows:
        issues.append({
            "type": "row_count",
            "message": f"Row count mismatch: {label1} has {table1.num_rows} rows, {label2} has {table2.num_rows} rows"
        })

    # Check columns
    cols1 = set(table1.column_names)
    cols2 = set(table2.column_names)

    cols_only_in_1 = cols1 - cols2
    cols_only_in_2 = cols2 - cols1
    common_cols = cols1 & cols2

    if cols_only_in_1:
        issues.append({
            "type": "columns",
            "message": f"{label1} has additional columns: {sorted(cols_only_in_1)}"
        })

    if cols_only_in_2:
        issues.append({
            "type": "columns",
            "message": f"{label2} has additional columns: {sorted(cols_only_in_2)}"
        })

    # Compare common columns (only if both tables have rows)
    if common_cols and table1.num_rows > 0 and table2.num_rows > 0:
        # Find a sortable column for comparison
        sort_column = None
        for col_name in sorted(common_cols):
            col_type = table1.schema.field(col_name).type
            if not pa.types.is_nested(col_type):
                sort_column = col_name
                break

        if sort_column is None:
            issues.append({
                "type": "sorting",
                "message": "No sortable column found in common columns - cannot compare row-by-row"
            })
        else:
            print(f"\nSorting by column: {sort_column}")
            try:
                table1_sorted = table1.sort_by(sort_column)
                table2_sorted = table2.sort_by(sort_column)

                print(f"\nComparing {len(common_cols)} common columns...")
                for col_name in sorted(common_cols):
                    print(f"  Checking {col_name}...", end=" ")
                    col1 = table1_sorted[col_name]
                    col2 = table2_sorted[col_name]

                    # For nested types (struct, list), use Python comparison
                    # PyArrow's .equals() checks exact representation, not logical equality
                    col_type = table1_sorted.schema.field(col_name).type
                    if pa.types.is_nested(col_type):
                        # Compare as Python objects for nested types
                        columns_equal = col1.combine_chunks().to_pylist() == col2.combine_chunks().to_pylist()
                    else:
                        # Use Arrow's equals for simple types
                        columns_equal = col1.equals(col2)

                    if not columns_equal:
                        print("MISMATCH")
                        # Find first difference
                        try:
                            # For nested types, find first difference manually
                            if pa.types.is_nested(col_type):
                                py1 = col1.to_pylist()
                                py2 = col2.to_pylist()
                                first_diff_idx = next(i for i in range(len(py1)) if py1[i] != py2[i])
                            else:
                                first_diff_idx = pc.index(pc.not_equal(col1, col2), True).as_py()
                            issues.append({
                                "type": "column_values",
                                "message": f"Column '{col_name}' has differences (first at row {first_diff_idx})"
                            })
                        except:
                            issues.append({
                                "type": "column_values",
                                "message": f"Column '{col_name}' has differences"
                            })
                    else:
                        print("OK")
            except Exception as e:
                issues.append({
                    "type": "sorting",
                    "message": f"Error during sorting/comparison: {str(e)}"
                })

    # Print final report
    print(f"\n{'='*70}")
    print(f"FINAL REPORT")
    print(f"{'='*70}")

    if not issues:
        print("✓ Tables are identical (possibly reordered)")
        return True
    else:
        print(f"✗ Found {len(issues)} difference(s):\n")

        # Group issues by type
        for idx, issue in enumerate(issues, 1):
            print(f"{idx}. [{issue['type'].upper()}] {issue['message']}")

        print(f"\n{'='*70}")
        return False


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Compare two PyArrow tables from parquet files or datasets directories",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare a transformed parquet with a datasets directory
  python compare.py data/transformed.parquet data/datasets_output

  # Compare two parquet files
  python compare.py output1.parquet output2.parquet

  # Compare two datasets directories
  python compare.py data/dataset1 data/dataset2
        """,
    )
    parser.add_argument(
        "file1", type=str, help="First file (parquet) or directory (datasets)"
    )
    parser.add_argument(
        "file2", type=str, help="Second file (parquet) or directory (datasets)"
    )
    args = parser.parse_args()

    # Load both tables
    print(f"Loading first table from: {args.file1}")
    table1 = load_table(args.file1)

    print(f"Loading second table from: {args.file2}")
    table2 = load_table(args.file2)

    # Compare tables and show full report
    compare_tables(table1, table2, label1=args.file1, label2=args.file2)


if __name__ == "__main__":
    main()
