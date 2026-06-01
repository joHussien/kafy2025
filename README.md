# KAFY: An Extensible and Scalable Transformers-Based System for Trajectory Data Analysis

This is the GitHub repository for the paper **"KAFY: An Extensible and Scalable Transformers-Based System for Trajectory Data Analysis"** by Youssef Hussein and Mohamed F. Mokbel. Submitted to VLDB 2026.

## Datasets

The experiments in the paper are conducted using two real-world GPS trajectory datasets: **Jakarta (Grab-Posisi)** and **GeoLife**. Below are detailed instructions on how to obtain each dataset, along with references to their original sources.

### 1. Jakarta (Grab-Posisi)

A ride-sharing dataset with 56K trajectories (56M GPS points) covering 500K km. In the paper, this dataset is used for the first four trajectory operations; namely, summarization, generation, prediction and imputation.

- **Source:** https://engineering.grab.com/grab-posisi
- **Access:** The dataset is available for **research purposes only** (non-commercial). To request access, email `grab.posisi@grabtaxi.com` with the following details:
  - Your name and contact details
  - Your institution
  - Your intended usage of the dataset
- **Format:** Each trajectory is serialized as an Apache Parquet file. Each GPS ping includes a trajectory ID, latitude, longitude, timestamp (UTC), accuracy level, bearing, and speed, sampled at a 1-second rate. The full dataset is approximately 2 GB.
- - **Citation:**
```bibtex
@inproceedings{huang2019grabposisi,
  author    = {Huang, Xiaocheng and Yin, Yifang and Lim, Simon and Wang, Guanfeng and Hu, Bo and Varadarajan, Jagannadan and Zheng, Shonali and Bulusu, Ali and Zimmermann, Roger},
  title     = {Grab-Posisi: An Extensive Real-Life GPS Trajectory Dataset in Southeast Asia},
  booktitle = {Proceedings of the 3rd ACM SIGSPATIAL International Workshop on Prediction of Human Mobility},
  pages     = {1--10},
  year      = {2019},
  doi       = {10.1145/3356995.3364536}
}
```
### 2. GeoLife

A dataset with 18K+ trajectories (24M GPS points) from 182 users covering 1.2M km. In the paper, this dataset is used for trajectory classification.

- **Source:** https://www.microsoft.com/en-us/research/publication/geolife-gps-trajectory-dataset-user-guide/
- **Access:** Publicly available for download from the Microsoft Research page above.
- **Description:** Collected by Microsoft Research Asia in the GeoLife project by 182 users over three years (April 2007 to August 2012). Each trajectory is a sequence of time-stamped points containing latitude, longitude, and altitude. The dataset contains 17,621 trajectories with a total distance of about 1.2 million kilometers and a total duration of 48,000+ hours, recorded at a variety of sampling rates (91% are densely logged, e.g., every 1–5 seconds or every 5–10 meters per point).
- **Citation:**
```bibtex
  @manual{zheng2011geolife,
    author  = {Zheng, Yu and Fu, Hao and Xie, Xing and Ma, Wei-Ying and Li, Quannan},
    title   = {Geolife GPS trajectory dataset - User Guide},
    year    = {2011},
    month   = {July},
    url     = {https://www.microsoft.com/en-us/research/publication/geolife-gps-trajectory-dataset-user-guide/},
    edition = {Geolife GPS trajectories 1.1}
  }
```
