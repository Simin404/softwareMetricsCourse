import logging
import requests
import csv
import time
import hashlib
import os
import json
from datetime import datetime, timedelta
import pandas as pd
import matplotlib.pyplot as plt

# -------------------- LOGGER SETUP --------------------
LOG_FILE = "information_quality.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",  # log pure JSON strings for easy parsing by other tools
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def log_event(event_type: str, **fields):
    """Emit a single structured JSON log line.
    `event_type` is a short string category; `fields` are arbitrary key/values.
    """
    payload = {
        "ts": datetime.utcnow().isoformat() + "Z",
        "event": event_type,
        **fields,
    }
    logger.info(json.dumps(payload, ensure_ascii=False))


# -------------------- CONFIGURATION --------------------
OUTPUT_FILE = "pytorch_issues_last6months.csv"
TOKEN = ""  # <-- Optional: add your GitHub personal access token
OWNER = "pytorch"
REPO = "pytorch"
SINCE_DATE = (datetime.utcnow() - timedelta(days=180)).isoformat() + "Z"
BASE_URL = f"https://api.github.com/repos/{OWNER}/{REPO}/issues"

HEADERS = {"Accept": "application/vnd.github.v3+json"}
if TOKEN:
    HEADERS["Authorization"] = f"token {TOKEN}"


# =========================================================
# =============== FILE INTEGRITY & ACCESS =================
# =========================================================

def is_file_accessible(path, mode='r'):
    try:
        with open(path, mode):
            pass
        return True
    except (IOError, PermissionError):
        return False


def get_file_hash(path):
    if not os.path.exists(path):
        return None
    hasher = hashlib.sha256()
    with open(path, 'rb') as f:
        while True:
            chunk = f.read(8192)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def check_file_integrity(file_path):
    section = "INTEGRITY_OF_OUTPUT_FILE"
    file_existed = os.path.exists(file_path)
    prev_hash = get_file_hash(file_path) if file_existed else None
    prev_mtime = os.path.getmtime(file_path) if file_existed else None

    if file_existed:
        accessible = is_file_accessible(file_path, 'r+')
        log_event(section, check="File writable", result="PASS" if accessible else "FAIL", path=file_path)
        if not accessible:
            raise PermissionError(f"File '{file_path}' is not accessible or writable.")
    else:
        writable_dir = os.access(os.path.dirname(os.path.abspath(file_path)) or ".", os.W_OK)
        log_event(section, check="Directory writable", result="PASS" if writable_dir else "FAIL", dir=os.path.dirname(os.path.abspath(file_path)) or ".")
        if not writable_dir:
            raise PermissionError(f"Cannot create file '{file_path}' – directory not writable.")

    if file_existed:
        time.sleep(0.1)
        current_hash = get_file_hash(file_path)
        current_mtime = os.path.getmtime(file_path)
        modified = (prev_hash != current_hash or prev_mtime != current_mtime)
        log_event(section, check="File modified externally", result="FAIL" if modified else "PASS", path=file_path, prev_hash=prev_hash, current_hash=current_hash, prev_mtime=prev_mtime, current_mtime=current_mtime)
        if modified:
            raise RuntimeError(f"File '{file_path}' was modified externally.")
    else:
        with open(file_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "number", "title", "state", "user", "created_at", "updated_at", "comments", "url"])
        log_event(section, check="File existence", result="NEW_FILE_CREATED", path=file_path)


# =========================================================
# =============== GITHUB ACCESS VALIDATION ================
# =========================================================
def validate_github_token_and_url(token: str, url: str):
    section = "ACCESSIBILITY_OF_DATA"
    # URL sanity
    if not url.startswith("https://api.github.com/repos/"):
        log_event(section, check="API URL format", result="FAIL", url=url)
        raise ValueError(f"Invalid GitHub API URL: {url}")
    log_event(section, check="API URL format", result="PASS", url=url)

    # Token presence & validity
    if not token:
        # Explicitly fail token-related check when no token is provided
        log_event(section, check="Token provided", result="FAIL", reason="No token set")
    else:
        tok_headers = {"Accept": "application/vnd.github.v3+json",
                       "Authorization": f"token {token}"}
        resp_tok = requests.get("https://api.github.com/rate_limit",
                                headers=tok_headers, timeout=10)
        if resp_tok.status_code == 401:
            log_event(section, check="Token validity", result="FAIL", status_code=resp_tok.status_code)
        elif resp_tok.status_code >= 400:
            log_event(section, check="Token validity", result="FAIL", status_code=resp_tok.status_code)
        else:
            log_event(section, check="Token validity", result="PASS", status_code=resp_tok.status_code)

    # API accessibility (independent of token presence) 
    hdrs = {"Accept": "application/vnd.github.v3+json"}
    if token:
        hdrs["Authorization"] = f"token {token}"
    resp_repo = requests.get(url, headers=hdrs, timeout=10)
    if resp_repo.status_code >= 400:
        log_event(section, check="API accessibility", result="FAIL", status_code=resp_repo.status_code)
        raise RuntimeError(f"GitHub API error {resp_repo.status_code}")
    else:
        log_event(section, check="API accessibility", result="PASS", status_code=resp_repo.status_code)


def pause_check(token: str):
    section = "ACCESSIBILITY_OF_DATA"

    headers = {"Accept": "application/vnd.github.v3+json"}
    if token:
        headers["Authorization"] = f"token {token}"

    resp = requests.get("https://api.github.com/rate_limit", headers=headers, timeout=10)
    data = resp.json()
    remaining = data["rate"]["remaining"]
    reset_time = int(data["rate"]["reset"])

    if remaining == 0:
        wait_seconds = max(0, reset_time - int(time.time()))
        log_event(section, check="Rate limit availability", result="WAITING", remaining=remaining, wait_seconds=wait_seconds)
        time.sleep(wait_seconds + 1)
    else:
        log_event(section, check="Rate limit availability", result="PASS", remaining=remaining)


# =========================================================
# =============== DATA QUALITY FUNCTIONS ==================
# =========================================================

def load_existing_issues(file_path):
    if not os.path.exists(file_path):
        return []
    with open(file_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [r for r in reader]


def get_latest_issue_date(issues):
    if not issues:
        return None
    try:
        return max(datetime.fromisoformat(i["created_at"].replace("Z", "+00:00")) for i in issues)
    except Exception:
        return None


def check_duplicates(issues):
    ids = [i.get("id") for i in issues if "id" in i]
    return list({i for i in ids if ids.count(i) > 1})


def fetch_new_issues(since_date, existing_ids):
    section = "FETCH_NEW_ISSUES"
    issues, page = [], 1
    params = {"state": "all", "since": since_date, "per_page": 100, "page": page}
    total_added = 0
    while True:
        resp = requests.get(BASE_URL, headers=HEADERS, params=params, timeout=15)
        if resp.status_code != 200:
            log_event(section, check="HTTP", result="FAIL", status_code=resp.status_code, page=page)
            break
        data = resp.json()
        if not data:
            log_event(section, check="Page empty", result="PASS", page=page)
            break
        added_this_page = 0
        for issue in data:
            if "pull_request" in issue:
                continue
            if issue["id"] not in existing_ids:
                issues.append({
                    "id": issue["id"],
                    "number": issue["number"],
                    "title": issue["title"],
                    "state": issue["state"],
                    "user": issue["user"]["login"],
                    "created_at": issue["created_at"],
                    "updated_at": issue["updated_at"],
                    "comments": issue["comments"],
                    "url": issue["html_url"]
                })
                added_this_page += 1
        total_added += added_this_page
        log_event(section, check="Page processed", result="PASS", page=page, items_returned=len(data), items_added=added_this_page)
        if len(data) < 100:
            break
        page += 1
        params["page"] = page
    log_event(section, check="Summary", result="PASS", total_added=total_added)
    return issues


def append_issues_to_csv(file_path, new_issues):
    section = "FETCH_NEW_ISSUES"
    if not new_issues:
        log_event(section, check="Append", result="SKIP", reason="No new issues")
        return
    with open(file_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=new_issues[0].keys())
        if os.stat(file_path).st_size == 0:
            writer.writeheader()
        writer.writerows(new_issues)
    log_event(section, check="Append", result="PASS", appended=len(new_issues), path=file_path)


# =========================================================
# =============== PREDICTION AND VISUALIZATION ============
# =========================================================

def plot_issues(df, title, include_prediction=False):
    weeks = df['Week']
    plt.figure()
    plt.plot(weeks, df['Open Issues'], label='Open Issues')
    plt.plot(weeks, df['Closed Issues'], label='Closed Issues')
    plt.xticks(rotation=45)
    if include_prediction:
        plt.axvline(x=weeks.iloc[-1], linestyle='--', alpha=0.7)
    plt.title(title)
    plt.xlabel('Week')
    plt.ylabel('Number of Issues')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    # Save a copy for auditability
    out_path = "weekly_open_closed_issues.png"
    plt.savefig(out_path)
    return out_path


def run_prediction_checks(output_file):
    section = "PREDICTION_CHECKS"
    reader = list(csv.DictReader(open(output_file, "r", encoding="utf-8")))
    today = datetime.utcnow().date()
    monday_this_week = today - timedelta(days=today.weekday())
    start_date = monday_this_week - timedelta(weeks=7)

    weekly_counts = []
    for i in range(8):
        week_start = start_date + timedelta(weeks=i)
        week_end = min(week_start + timedelta(days=6), today)
        weekly_counts.append({"Week": f"{week_start} → {week_end}", "Open Issues": 0, "Closed Issues": 0})

    for issue in reader:
        try:
            created_at = datetime.fromisoformat(str(issue["created_at"]).replace("Z", "+00:00")).date()
        except Exception:
            continue
        state = str(issue.get("state", "")).lower()
        for week in weekly_counts:
            week_start_str, week_end_str = week["Week"].split(" → ")
            ws = datetime.strptime(week_start_str, "%Y-%m-%d").date()
            we = datetime.strptime(week_end_str, "%Y-%m-%d").date()
            if ws <= created_at <= we:
                if state == "closed":
                    week["Closed Issues"] += 1
                else:
                    week["Open Issues"] += 1
                break

    df_weeks = pd.DataFrame(weekly_counts)
    last_2 = df_weeks.tail(2)
    predicted_open = last_2['Open Issues'].mean()
    predicted_closed = last_2['Closed Issues'].mean()

    warning_conditions = {
        "A": "Predicted open > 80 or predicted closed < 30",
        "B": "Difference > 90",
        "C": "Open increasing, Closed decreasing"
    }

    triggered = []
    if predicted_open > 80 or predicted_closed < 30:
        triggered.append("A")
    if (predicted_open - predicted_closed) > 90:
        triggered.append("B")
    if (predicted_open > last_2['Open Issues'].iloc[-1]) and (predicted_closed < last_2['Closed Issues'].iloc[-1]):
        triggered.append("C")

    new_row = pd.DataFrame([{
        "Week": "Next Week",
        "Open Issues": round(predicted_open, 2),
        "Closed Issues": round(predicted_closed, 2),
        "Warning_type": ",".join(triggered) if triggered else None
    }])
    df_weeks = pd.concat([df_weeks, new_row], ignore_index=True)

    # Emit logs for each computed signal
    log_event(section, check="Predicted values", result="PASS", predicted_open=round(predicted_open, 2), predicted_closed=round(predicted_closed, 2))
    log_event(section, check="Warning thresholds", result=("FAIL" if triggered else "PASS"), warnings=triggered or []
)

    # Visualization
    try:
        img_path = plot_issues(df_weeks, "Weekly Open/Closed Issues with Prediction", include_prediction=True)
        plt.close()
        log_event(section, check="Visualization of summarized data", result="PASS", image_path=img_path)
    except Exception as e:
        log_event(section, check="Visualization of summarized data", result="FAIL", error=str(e))


# =========================================================
# ===================== MAIN EXECUTION ====================
# =========================================================

def main():
    log_event("RUN", stage="START", message="Starting integrity and quality checks")

    try:
        check_file_integrity(OUTPUT_FILE)
        validate_github_token_and_url(TOKEN, BASE_URL)
        pause_check(TOKEN)
    except Exception as e:
        log_event("RUN", stage="ABORT", error=str(e))
        return

    existing = load_existing_issues(OUTPUT_FILE)
    existing_ids = {int(i["id"]) for i in existing if i.get("id")}
    latest_date = get_latest_issue_date(existing)
    since_date = latest_date.isoformat().replace("+00:00", "Z") if latest_date else SINCE_DATE

    log_event("FETCH_NEW_ISSUES", since_date=since_date, existing_count=len(existing))

    new_issues = fetch_new_issues(since_date, existing_ids)
    append_issues_to_csv(OUTPUT_FILE, new_issues)

    # Data quality checks
    duplicates = check_duplicates(existing)
    log_event("DATA_QUALITY_CHECKS", check="Duplicate issue IDs", result="PASS" if not duplicates else "FAIL", duplicates=duplicates)

    log_event("DATA_QUALITY_CHECKS", check="New issues fetched", result="PASS", count=len(new_issues))

    total_unique = len(set(i["id"] for i in existing + new_issues))
    consistency = "FAIL" if (len(existing) + len(new_issues)) > total_unique else "PASS"
    log_event("DATA_QUALITY_CHECKS", check="Missing record consistency", result=consistency, total_records=len(existing) + len(new_issues), total_unique=total_unique)

    run_prediction_checks(OUTPUT_FILE)

    log_event("RUN", stage="END", message="All quality checks completed successfully")


if __name__ == "__main__":
    main()
