import json
import logging

logger = logging.getLogger(__name__)
def validate_config(config, schema, prefix="") -> list:
    errors = []
    for key, expected_type in schema.items():
        full_key = f"{prefix}.{key}" if prefix else key
        if key not in config:
            errors.append(f"Missing key: {full_key}")
        else:
            value = config[key]
            if isinstance(expected_type, dict):
                if not isinstance(value, dict):
                    errors.append(f"{full_key} should be a dict")
                else:
                    errors += validate_config(value, expected_type, prefix=full_key)
            else:
                if not isinstance(value, expected_type):
                    errors.append(f"{full_key} should be {expected_type.__name__}, got {type(value).__name__}")

    # Warn about unexpected keys
    for key in config:
        if key not in schema:
            full_key = f"{prefix}.{key}" if prefix else key
            errors.append(f"Unexpected key: {full_key}")

    return errors


def pprint_errors(err_list:list) -> None:
    for err in err_list:
        logger.info(" %s", err)
