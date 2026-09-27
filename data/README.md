# Dataset

The project currently uses:

```text
data_penjualan.csv
```

Required columns:

```text
stok,terjual,hari_ke,cuaca,hari_besar,sisa_persen,label
```

Column usage in the current project:

- `stok` = available stock
- `terjual` = items sold
- `hari_ke` = day index in the dataset
- `cuaca` = weather code (`0` or `1`)
- `hari_besar` = special-day code (`0` or `1`)
- `sisa_persen` = remaining stock percentage
- `label` = prediction target

Label values:

- `0` = Safe
- `1` = Potential Waste

## Important Note

The original source of this dataset is not documented in the supplied project files. The meanings of weather codes `0` and `1` are also not documented, so this project keeps those values as raw codes instead of inventing labels.

If the original dataset source is known, add the source link or citation here before publishing the project.
