# ASX/ABS Early Warning Platform



A production-style financial risk analytics prototype that combines ASX market signals with macroeconomic context to identify early warning indicators of company or sector-level stress.



> Status: active portfolio upgrade. The project currently includes a FastAPI scoring service, Streamlit dashboard, trained model artifacts, cached ASX market data, evaluation outputs, and prototype risk-scoring workflows.



---



## Why this project matters



Financial institutions, analysts, and risk teams need early indicators of market stress before distress becomes obvious in headline results. This project explores whether market-derived features such as returns, volatility, drawdown, momentum, and liquidity proxies can be used to generate an early warning risk score for ASX-listed companies.



This project is designed as a portfolio-grade system, not just a notebook. It includes:



- data ingestion and feature generation

- trained ML artifacts

- FastAPI prediction endpoints

- Streamlit dashboard

- model metrics and figures

- reproducible local setup

- clear limitations and roadmap



---



## Current capability



The current implementation can:



- load processed ASX market feature data

- score a single company through a FastAPI endpoint

- score batches of tickers through `/predict_batch`

- display a Streamlit risk-scoring dashboard

- show model threshold and risk probabilities

- use cached Yahoo Finance market data

- load trained model artifacts from `models/` or fallback artifacts from `artifacts/`

- surface baseline evaluation outputs and model cards



---



## Architecture



```text

ASX / Yahoo Finance data

        |

        v

data/cache/yahoo/\*.parquet

        |

        v

src/market_data.py

src/price_features.py

src/build_dataset.py

        |

        v

data/processed/market_firm_features.csv

        |

        v

src/train.py

        |

        v

models/model.joblib

models/metrics.json

        |

        +-----------------------------+

        |                             |

        v                             v

FastAPI scoring service          Streamlit dashboard

app/api.py                       app/ui.py

        |                             |

        v                             v

/predict, /predict_batch          single company scoring

/health, /docs                    top-risk exploration

