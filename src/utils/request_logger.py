import pandas as pd
import os
import threading

LOG_PATH= "logs/prediction_requests.csv"
_lock=threading.Lock()

def log_request(payload: dict):
    os.makedirs(os.path.dirname(LOG_PATH),exist_ok=True)
    df_row=pd.DataFrame([payload])
    with _lock:
        write_header=not os.path.exists(LOG_PATH)
        df_row.to_csv(LOG_PATH, mode='a', header=write_header, index=False)

def load_recent_requests(n: int=1000)->pd.DataFrame:
    if not os.path.exists(LOG_PATH):
        return pd.DataFrame()
    df=pd.read_csv(LOG_PATH)
    return df.tail(n)
