import logging 
import json
from datetime import datetime, timezone

class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_obj={
            "timestamp":datetime.now(timezone.utc).isoformat(),
            "level":record.levelname,
            "message":record.getMessage(),
        }
        if hasattr(record,"extra_data"):
            log_obj.update(record.extra_data)
        return json.dumps(log_obj)

def get_logger(name: str):
    logger=logging.getLogger(name)
    if not logger.handlers:
        handler=logging.StreamHandler()
        handler.setFormatter(JSONFormatter())
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger
