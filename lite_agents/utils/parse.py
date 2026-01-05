from typing import Any
from json_repair import repair_json
import re
import json
from lite_agents.logger import setup_logger

logger = setup_logger()

def parse_json_from_keys(text: str, keys: str | list[str]) -> dict[str, Any] | None:
    """Parse raw text based on the provided JSON keys.

    Args:
        text (str): raw text.
        keys (str | list[str]): keys to extract from the JSON.

    Returns:
        dict[str, Any] | None: The parsed data as a dictionary or None if not found.
    """

    if isinstance(keys, str):
        keys = [keys]

    json_pattern = r"\{[\s\S]*?\}"
    matches = re.findall(json_pattern, text)

    if not matches:
        logger.warning(f"No JSON structure found in the text: {text}")
        return None

    raw_data = matches[0]

    try:
        raw_data = repair_json(raw_data)
        parsed_data = json.loads(raw_data)
    except Exception as e:
        logger.error(f"Error parsing data with JSON: {e}")
        return None

    # Extract relevant keys from parsed_data
    result = {}
    for key in keys:
        if key in parsed_data:
            result[key] = parsed_data[key]
        else:
            logger.warning(f"Key {key} not found in the JSON data")
    return result
