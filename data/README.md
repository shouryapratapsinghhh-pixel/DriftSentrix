# Data

Not required to run tests or `make_synthetic` -- only needed for `load_smd` / `load_nasa`.

## SMD (Server Machine Dataset)
Source: https://github.com/NetManAIOps/OmniAnomaly (data/ folder)
Expected layout:
```
data/smd/train/<machine>.txt
data/smd/test/<machine>.txt
data/smd/test_label/<machine>.txt
```

## SMAP/MSL (NASA telemetry)
Source: https://github.com/khundman/telemanom (labeled_anomalies.csv + data/)
Expected layout:
```
data/nasa/train/<channel>.npy
data/nasa/test/<channel>.npy
data/nasa/labeled_anomalies.csv
```
