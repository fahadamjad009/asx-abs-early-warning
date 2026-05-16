# ASX/ABS Early Warning Platform — Architecture



## Purpose



This project is a production-style financial risk analytics prototype for ASX-listed companies. It combines market-derived indicators, trained model artifacts, a FastAPI scoring service, and a Streamlit dashboard into a local end-to-end risk scoring system.



The current system focuses on market-derived early warning signals. ABS/RBA macroeconomic feature integration is planned as the next major modelling extension.



---



## System flow



```text

ASX / Yahoo Finance market data

&#x20;       |

&#x20;       v

data/cache/yahoo/\*.parquet

&#x20;       |

&#x20;       v

src/market\_data.py

&#x20;       |

&#x20;       v

src/price\_features.py

&#x20;       |

&#x20;       v

src/build\_dataset.py

&#x20;       |

&#x20;       v

data/processed/market\_firm\_features.csv

&#x20;       |

&#x20;       v

src/train.py

&#x20;       |

&#x20;       v

models/model.joblib

models/metrics.json

&#x20;       |

&#x20;       +-------------------------------+

&#x20;       |                               |

&#x20;       v                               v

app/api.py                         app/ui.py

FastAPI scoring service             Streamlit dashboard

&#x20;       |                               |

&#x20;       v                               v

/predict, /predict\_batch             single ticker scoring

/health, /docs                       dashboard exploration

