import sys

def tokenize_mask_extract_and_apply(
    ds, model, tok, col_name, args, idx):

    batch_size = args.max_batch_size

    ds = ds.map(
        lambda x: {col_name: process_spaces(x[col_name]).replace("\n", " ")})

    ds = ds.map(
        lambda x: {f"modification_{idx}":
            tokenize_and_mask(x[col_name], span_length=2, pct=args.pct_mask)},
        batched=False, desc = "Tokenizing and masking")

    ds = ds.map(
        lambda x: {f"fills_{idx}":
            replace_masks(x[f"modification_{idx}"], model, tok)},
        batched=True, batch_size=batch_size, desc = "Replacing masks")

    ds = ds.map(
        lambda x: {f"fills_{idx}":
            extract_fills(x[f"fills_{idx}"])},
        batched=True, batch_size=batch_size, desc = "Extracting fills")

    col_name = f"{col_name}_synthetic_{idx}"

    ds = ds.map(
        lambda x: {col_name:
            apply_extracted_fills(x[f"modification_{idx}"], x[f"fills_{idx}"])},
        batched=True, batch_size=batch_size, desc = "Applying extracted fills")

    ds = ds.remove_columns([f"fills_{idx}", f"modification_{idx}"])

    return ds, col_name