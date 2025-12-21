from datetime import datetime

from lite_agents.logger import setup_logger

logger = setup_logger()


def today(ddmmyy: bool = True) -> str:
    """Return the current date in the specified format.

    Args:
        ddmmyy (bool, optional): if True, return date in "Weekday DD-MM-YYYY" format. Defaults to True.

    Returns:
        str: formatted current date string.
    """
    today_date = datetime.now().date()

    if ddmmyy:
        return today_date.strftime("%A %d-%m-%Y")
    else:
        return today_date.strftime("%A %Y-%m-%d")


def validate_dates(start_date: str, final_date: str) -> tuple[bool, str | None]:
    """Easy validation date for a request with start and end date. Validation includes:

        - Both initial date and final date must be provided.
        - Initial date must be before or equal to final date.
        - Dates cannot be in the previous months or years

    Args:
        start_date (str): initial date in DD-MM-YYYY or YYYY-MM-DD format.
        final_date (str): final date in DD-MM-YYYY or YYYY-MM-DD format.

    Returns:
        tuple[bool, str | None]: validation result and error message if any
    """
    current_date = datetime.now().date()

    def parse_date(d_str: str) -> datetime.date:  # type: ignore
        """Parse date from a string.

        Args:
            d_str (str): date string in DD-MM-YYYY or YYYY-MM-DD format.

        Raises:
            ValueError: if the date string does not match expected formats.

        Returns:
            datetime.date: parsed date object
        """
        for fmt in ("%d-%m-%Y", "%Y-%m-%d"):
            try:
                return datetime.strptime(d_str, fmt).date()
            except ValueError:
                pass
        raise ValueError(f"Invalid date format: {d_str}")

    try:
        start_date_dt = parse_date(start_date)
        final_date_dt = parse_date(final_date)
    except ValueError:
        return False, "Formato data non valido. Usa DD-MM-YYYY o YYYY-MM-DD."

    # logger.info(
    #     f"Validating dates: current_date={datetime.now().date()}, start_date={start_date_dt}, final_date={final_date_dt}... (stub function)"
    # )
    # Stub validation logic
    if not start_date_dt or not final_date_dt:
        return False, "Both start date and final date must be provided."
    if start_date_dt > final_date_dt:  # type: ignore
        return (False, "Start date must be before or equal to end date.")
    if start_date_dt.year < current_date.year:  # type: ignore
        return (
            False,
            "Starting date cannot be in a month or year prior to the current one.",
        )
    if (
        start_date_dt.year == current_date.year  # type: ignore
        and start_date_dt.month < current_date.month  # type: ignore
    ):
        return (False, "Starting date cannot be in a month prior to the current one.")
    return True, None


def calculate_date(start_date: str, delta_days: int) -> str:
    """Given a start date and a delta in days, return the end date.

    Args:
        start_date (str): start date in DD-MM-YYYY or YYYY-MM-DD format
        delta_days (int): delta in days

    Returns:
        str: end date in DD-MM-YYYY format
    """
    from datetime import timedelta

    try:
        start_date_dt = datetime.strptime(start_date, "%d-%m-%Y").date()
    except ValueError:
        try:
            start_date_dt = datetime.strptime(start_date, "%Y-%m-%d").date()
        except ValueError:
            raise ValueError(
                f"Date format not recognized: {start_date}. Use DD-MM-YYYY or YYYY-MM-DD"
            )

    end_date = start_date_dt + timedelta(days=delta_days)
    # logger.info(
    #     f"Calculating end date: start_date={start_date}, delta_days={delta_days}, end_date={end_date}"
    # )
    return end_date.strftime("%d-%m-%Y")
