# Data Sources & BigQuery / Vertex AI Feature Store Schema Catalog

## 1. Primary & Backup Verified Public Datasets

| Dataset Role | Dataset Name | Direct URL | Scale & Granularity |
| :--- | :--- | :--- | :--- |
| **Primary Public Dataset (Smart Home Environmental & Appliance Telemetry)** | UCI Appliances Energy Prediction & ZigBee Wireless Sensor Network Telemetry | https://archive.ics.uci.edu/dataset/374/appliances+energy+prediction | `19,735` multivariate 10-minute IoT sensor timestamps × `29` environmental, weather, and appliance power channels |
| **Primary Streaming Anomaly Benchmark** | Numenta Anomaly Benchmark (NAB) Real-Time Cloud & IoT Telemetry Corpus | https://github.com/numenta/NAB | `365,551` streaming time-series sensor readings across `58` labeled anomaly streams (`385K+` combined IoT & telemetry records) |
| **Backup Public Dataset** | NASA Acoustics and Vibration Database / CMAPSS Degradation Sensor Streams | https://data.nasa.gov/Aerospace/CMAPSS-Jet-Engine-Simulated-Data/ff5v-kuh6 | `160,000+` multi-sensor time-series cycles across `21` physical telemetry channels |

---

## 2. BigQuery Warehouse & Vertex AI Feature Store Table Schema (`iot_telemetry_warehouse.sensor_engineered_features`)

| Column Name | BigQuery Data Type | Vertex AI Feature Store Role | Description & Engineering Contract |
| :--- | :--- | :--- | :--- |
| `device_id` | `STRING NOT NULL` | `entity_id` | Unique edge gateway / smart-sensor identifier (`DEV-1000` .. `DEV-9999`) |
| `event_timestamp` | `TIMESTAMP NOT NULL` | `feature_time` | Point-in-time UTC sensor observation timestamp (monotonicity enforced) |
| `temp_indoor_c` | `FLOAT64 NOT NULL` | Online / Offline Feature | Indoor ambient temperature (°C) from ZigBee thermistor node |
| `humidity_indoor_pct` | `FLOAT64 NOT NULL` | Online / Offline Feature | Relative indoor humidity (%) within `[0.0, 100.0]` |
| `power_draw_watts` | `FLOAT64 NOT NULL` | Online / Offline Feature | Instantaneous electrical load (Watts) |
| `vibration_rms_g` | `FLOAT64 NOT NULL` | Online / Offline Feature | Root-mean-square triaxial accelerometer vibration ($g$) |
| `pressure_hpa` | `FLOAT64 NOT NULL` | Online / Offline Feature | Barometric pressure (hPa) |
| `motion_pir_score` | `FLOAT64 NOT NULL` | Online / Offline Feature | Passive infrared occupancy/motion activation score `[0.0, 1.0]` |
| `acoustic_db` | `FLOAT64 NOT NULL` | Online / Offline Feature | Acoustic sound pressure level (dB SPL) |
| `power_draw_watts_roll_mean` | `FLOAT64` | Engineered Window Feature | Rolling $W$-window mean electrical power draw per `device_id` |
| `power_draw_watts_roll_std` | `FLOAT64` | Engineered Window Feature | Rolling $W$-window standard deviation of electrical load |
| `electro_vib_interaction` | `FLOAT64` | Cross-Sensor Feature | Product of `power_draw_watts * vibration_rms_g` capturing mechanical load |
| `acoustic_motion_product` | `FLOAT64` | Cross-Sensor Feature | Coupled acoustic-motion activation index |
| `thermal_humidity_ratio` | `FLOAT64` | Cross-Sensor Feature | Ratio of indoor temperature to relative humidity |
| `is_anomaly` | `INT64 NOT NULL` | Supervised Target Label | Binary ground-truth equipment fault / intrusion anomaly label (`0` or `1`) |
