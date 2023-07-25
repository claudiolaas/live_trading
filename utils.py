import json
import logging
import smtplib
import ssl
import subprocess
import sys
from pathlib import Path


def get_git_revision_hash() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip()


def get_git_revision_short_hash() -> str:
    return (
        subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
        .decode("ascii")
        .strip()
    )


def get_logger(log_type, market, config):
    logger = logging.getLogger(log_type)
    logger.setLevel(logging.DEBUG)

    log_pth = Path(config["log_pth"])

    if not log_pth.exists():
        log_pth.mkdir(parents=True)

    file_handler = logging.FileHandler(log_pth / f"{log_type}.log")
    formatter = logging.Formatter(
        "%(asctime)s : %(levelname)s : %(name)s : %(message)s"
    )
    file_handler.setFormatter(formatter)

    # add file handler to logger
    logger.addHandler(file_handler)
    logger.addHandler(logging.StreamHandler(sys.stdout))
    return logger


def send_email(subject, text):
    return
    port = 587  # For starttls
    smtp_server = "smtp.gmail.com"
    sender_email = "overfitting24@gmail.com"
    receiver_email = "overfitting24@gmail.com"

    message = f"Subject: {subject} \n \n {text}"
    context = ssl.create_default_context()
    with smtplib.SMTP(smtp_server, port) as server:
        server.ehlo()  # Can be omitted
        server.starttls(context=context)
        server.ehlo()  # Can be omitted
        server.login(sender_email, "endlichreich")
        server.sendmail(sender_email, receiver_email, message)


def save_config(config: dict, save_path: Path = None) -> None:
    if save_path is None:
        save_path = Path(config["log_pth"])
    if not save_path.exists():
        save_path.mkdir(parents=True)
    with open(save_path / "config.json", "w+") as fout:
        json.dump(config, fout, indent=2)


def read_credentials() -> dict:
    cred_path = Path("credentials.json")
    if not cred_path.exists():
        print(f"WARN: credentials file {str(cred_path)} doesn't exist.")
        return {}
    with open(cred_path, "r") as fin:
        creds = json.load(fin)
    return creds


def str2bool(s: str) -> bool:
    if s.lower() == "true":
        return True
    elif s.lower() == "false":
        return False
    else:
        raise RuntimeError("Boolean value expected")


def truncate2(f: float, n: int = 8) -> float:
    """Truncates/pads a float f to n decimal places without rounding
    https://stackoverflow.com/a/783927
    https://stackoverflow.com/a/22155830"""
    s = f"{f}"
    if "e" in s or "E" in s:
        return "{0:.{1}f}".format(f, n)
    i, p, d = s.partition(".")
    return ".".join([i, (d + "0" * n)[:n]])
