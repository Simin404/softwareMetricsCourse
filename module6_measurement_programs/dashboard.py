# Setup:
#   pip install streamlit pandas requests python-dateutil
#   export GITHUB_TOKEN=""   # recommended to avoid low rate limits
# Run:
#   streamlit run dashboard.py

import os
import time
import math
import requests
import pandas as pd
import streamlit as st
from datetime import datetime, timedelta, timezone
from dateutil import parser as dateparser

GITHUB_API = "https://api.github.com"


def utc_now():
    return datetime.now(timezone.utc)


def iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def to_dt(s: str) -> datetime:
    return dateparser.isoparse(s).astimezone(timezone.utc)


class GitHubClient:
    def __init__(self, token: str | None):
        self.s = requests.Session()
        self.s.headers.update(
            {
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
            }
        )
        if token:
            self.s.headers.update({"Authorization": f"Bearer {token}"})

    def _request(self, method: str, url: str, params=None, headers=None, retry=3):
        for attempt in range(retry):
            r = self.s.request(method, url, params=params, headers=headers, timeout=30)
            if r.status_code == 403 and r.headers.get("X-RateLimit-Remaining") == "0":
                reset = int(r.headers.get("X-RateLimit-Reset", "0"))
                sleep_s = max(1, reset - int(time.time()) + 1)
                time.sleep(min(sleep_s, 60))
                continue
            if r.status_code in (500, 502, 503, 504):
                time.sleep(1.5 * (attempt + 1))
                continue
            r.raise_for_status()
            return r
        r.raise_for_status()

    def get(self, path: str, params=None, headers=None):
        return self._request("GET", f"{GITHUB_API}{path}", params=params, headers=headers)

    def paginate(self, path: str, params=None, headers=None, item_key=None, max_pages=20):
        """
        Generic paginator for GitHub REST.
        If response is list, yields items from list.
        If response is dict, uses item_key to yield items from dict[item_key].
        """
        params = dict(params or {})
        params.setdefault("per_page", 100)
        page = 1
        while page <= max_pages:
            params["page"] = page
            r = self.get(path, params=params, headers=headers)
            data = r.json()
            if isinstance(data, list):
                items = data
            else:
                if not item_key:
                    raise ValueError("item_key required for dict responses")
                items = data.get(item_key, [])
            if not items:
                break
            for it in items:
                yield it
            page += 1


def day_bucket(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%d")


def series_by_day(datetimes: list[datetime], start: datetime, end: datetime) -> pd.DataFrame:
    days = pd.date_range(start=start.date(), end=end.date(), freq="D", tz="UTC")
    counts = pd.Series(0, index=days)
    for dt in datetimes:
        d = pd.Timestamp(dt.date(), tz="UTC")
        if d in counts.index:
            counts.loc[d] += 1
    df = counts.rename("count").to_frame()
    df.index.name = "day"
    return df.reset_index()


def median_p90(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    s = sorted(values)
    n = len(s)
    med = s[n // 2] if n % 2 == 1 else 0.5 * (s[n // 2 - 1] + s[n // 2])
    p90 = s[int(math.ceil(0.9 * n)) - 1]
    return med, p90


def compute_time_to_fix_broken_build(workflow_runs: list[dict]) -> dict:
    """
    Approximates "time to fix a broken build" from GitHub Actions workflow runs on default branch.
    Method:
      - Consider only completed runs with a conclusion.
      - Detect failure streak start, and when next success occurs, record fix duration.
      - Duration measured from first failing run completed_at to success run completed_at.
    """
    relevant = []
    for r in workflow_runs:
        if r.get("status") != "completed":
            continue
        if not r.get("conclusion"):
            continue
        if not r.get("completed_at") or not r.get("created_at"):
            continue
        relevant.append(r)

    relevant.sort(key=lambda x: x["created_at"])

    failing = {"failure", "timed_out", "cancelled", "action_required", "startup_failure"}
    fix_times_hours = []

    in_break = False
    break_start_completed = None

    for r in relevant:
        concl = r["conclusion"]
        completed = to_dt(r["completed_at"])

        if not in_break and concl in failing:
            in_break = True
            break_start_completed = completed
            continue

        if in_break and concl == "success":
            delta = completed - break_start_completed
            fix_times_hours.append(delta.total_seconds() / 3600.0)
            in_break = False
            break_start_completed = None

    med, p90 = median_p90(fix_times_hours)
    return {
        "samples": len(fix_times_hours),
        "median_hours": med,
        "p90_hours": p90,
        "all_hours": fix_times_hours,
    }


@st.cache_data(ttl=900)
def fetch_repo_bundle(repo_full: str, days: int, token_present: bool):
    token = os.environ.get("GITHUB_TOKEN") if token_present else None
    gh = GitHubClient(token)

    owner, repo = repo_full.split("/")
    end = utc_now()
    start = end - timedelta(days=days)
    start_iso = iso(start)

    # Repo info (stars, forks, default branch)
    repo_info = gh.get(f"/repos/{owner}/{repo}").json()
    default_branch = repo_info.get("default_branch", "main")

    # Workflow runs on default branch since start
    runs = []
    for item in gh.paginate(
        f"/repos/{owner}/{repo}/actions/runs",
        params={"branch": default_branch, "created": f">={start_iso}"},
        item_key="workflow_runs",
        max_pages=20,
    ):
        runs.append(item)

    # Commits since start
    commits = []
    for c in gh.paginate(
        f"/repos/{owner}/{repo}/commits",
        params={"since": start_iso},
        max_pages=20,
    ):
        commits.append(c)

    # Search issues and PRs since start
    # Using Search API gives created/closed times and is straightforward for per-day counts.
    def search_all(query: str, max_pages=10):
        items = []
        for page in range(1, max_pages + 1):
            r = gh.get(
                "/search/issues",
                params={"q": query, "per_page": 100, "page": page},
            ).json()
            batch = r.get("items", [])
            if not batch:
                break
            items.extend(batch)
            # Search API returns up to 1000 results. Stop if we are near the cap.
            if page * 100 >= 1000:
                break
        return items

    date_q = start.date().isoformat()
    prs_opened = search_all(f"repo:{repo_full} is:pr created:>={date_q}", max_pages=10)
    issues_opened = search_all(f"repo:{repo_full} is:issue created:>={date_q}", max_pages=10)
    issues_closed = search_all(f"repo:{repo_full} is:issue closed:>={date_q}", max_pages=10)

    # Recent stargazers for stars gained, and recent forks for forks gained
    # These can be heavy for large repos. We cap pages and stop once older than the window.
    stars_gained = []
    stargazer_headers = {"Accept": "application/vnd.github.star+json"}

    for sg in gh.paginate(
        f"/repos/{owner}/{repo}/stargazers",
        params={},
        headers=stargazer_headers,
        max_pages=30,  # adjust in UI if needed
    ):
        starred_at = sg.get("starred_at")
        if not starred_at:
            continue
        dt = to_dt(starred_at)
        if dt < start:
            break
        stars_gained.append(dt)

    forks_gained = []
    for fk in gh.paginate(
        f"/repos/{owner}/{repo}/forks",
        params={"sort": "newest"},
        max_pages=10,
    ):
        created = fk.get("created_at")
        if not created:
            continue
        dt = to_dt(created)
        if dt < start:
            break
        forks_gained.append(dt)

    return {
        "repo_info": repo_info,
        "default_branch": default_branch,
        "start": start,
        "end": end,
        "runs": runs,
        "commits": commits,
        "prs_opened": prs_opened,
        "issues_opened": issues_opened,
        "issues_closed": issues_closed,
        "stars_gained": stars_gained,
        "forks_gained": forks_gained,
    }


def main():
    st.set_page_config(page_title="GitHub CI and Dev Metrics Dashboard", layout="wide")

    st.title("GitHub Repository Metrics Dashboard")

    with st.sidebar:
        st.header("Settings")
        repo_full = st.text_input("Repository (owner/name)", value="pytorch/pytorch")
        days = st.slider("Window (days)", min_value=7, max_value=90, value=30, step=1)
        use_token = st.checkbox("Use GITHUB_TOKEN from environment", value=True)
        st.caption("Tip: set GITHUB_TOKEN to increase API rate limits.")

    bundle = fetch_repo_bundle(repo_full, days, use_token)

    repo_info = bundle["repo_info"]
    start = bundle["start"]
    end = bundle["end"]

    st.subheader(f"{repo_full}  |  {bundle['default_branch']}  |  {start.date()} to {end.date()}")

    # Top KPIs
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Stars (total)", f"{repo_info.get('stargazers_count', 0):,}")
    col2.metric("Forks (total)", f"{repo_info.get('forks_count', 0):,}")
    col3.metric("Open issues", f"{repo_info.get('open_issues_count', 0):,}")
    col4.metric("Commits (window)", f"{len(bundle['commits']):,}")
    col5.metric("Workflow runs (window)", f"{len(bundle['runs']):,}")

    st.divider()

    # Metric 1: Time to fix a broken build
    st.header("CI Metrics")

    fix_stats = compute_time_to_fix_broken_build(bundle["runs"])
    c1, c2, c3 = st.columns(3)
    c1.metric("Time to fix broken build, median (hours)", "n/a" if fix_stats["median_hours"] is None else f"{fix_stats['median_hours']:.2f}")
    c2.metric("Time to fix broken build, p90 (hours)", "n/a" if fix_stats["p90_hours"] is None else f"{fix_stats['p90_hours']:.2f}")
    c3.metric("Fix samples", f"{fix_stats['samples']}")

    # Metric 2: frequency/volume of CI builds over time
    run_created = [to_dt(r["created_at"]) for r in bundle["runs"] if r.get("created_at")]
    df_runs = series_by_day(run_created, start, end)
    st.subheader("CI build volume (workflow runs per day)")
    st.line_chart(df_runs.set_index("day")["count"])

    # Metric 3: CI build duration
    durations_min = []
    dur_points = []
    for r in bundle["runs"]:
        if r.get("run_started_at") and r.get("updated_at"):
            a = to_dt(r["run_started_at"])
            b = to_dt(r["updated_at"])
            if b >= a:
                dur = (b - a).total_seconds() / 60.0
                durations_min.append(dur)
                dur_points.append({"day": pd.Timestamp(a.date(), tz="UTC"), "duration_min": dur})
        elif r.get("created_at") and r.get("updated_at"):
            a = to_dt(r["created_at"])
            b = to_dt(r["updated_at"])
            if b >= a:
                dur = (b - a).total_seconds() / 60.0
                durations_min.append(dur)
                dur_points.append({"day": pd.Timestamp(a.date(), tz="UTC"), "duration_min": dur})

    med_dur, p90_dur = median_p90(durations_min)
    c1, c2 = st.columns(2)
    c1.metric("CI build duration, median (min)", "n/a" if med_dur is None else f"{med_dur:.1f}")
    c2.metric("CI build duration, p90 (min)", "n/a" if p90_dur is None else f"{p90_dur:.1f}")

    if dur_points:
        df_dur = pd.DataFrame(dur_points)
        df_dur = df_dur.groupby("day", as_index=False)["duration_min"].median().rename(columns={"duration_min": "median_duration_min"})
        st.subheader("CI build duration (median per day, minutes)")
        st.line_chart(df_dur.set_index("day")["median_duration_min"])
    else:
        st.info("No workflow duration data available in this window (or API fields missing).")

    st.divider()

    st.header("Development Activity Metrics")

    # Metric 4: Commit activity
    commit_dts = []
    contributors = set()
    for c in bundle["commits"]:
        cd = c.get("commit", {}).get("committer", {}).get("date") or c.get("commit", {}).get("author", {}).get("date")
        if cd:
            commit_dts.append(to_dt(cd))
        # contributor identity
        if c.get("author") and c["author"].get("login"):
            contributors.add(c["author"]["login"])
        else:
            email = c.get("commit", {}).get("author", {}).get("email")
            if email:
                contributors.add(email.lower())

    df_commits = series_by_day(commit_dts, start, end)
    st.subheader("Commit activity (commits per day)")
    st.line_chart(df_commits.set_index("day")["count"])

    # Metric 5: Active contributors
    st.metric("Active contributors (unique, window)", f"{len(contributors):,}")

    # Metric 6: PRs opened volume
    pr_created = [to_dt(it["created_at"]) for it in bundle["prs_opened"] if it.get("created_at")]
    df_prs = series_by_day(pr_created, start, end)
    st.subheader("Pull requests opened (per day)")
    st.line_chart(df_prs.set_index("day")["count"])

    # Metric 7: Issues opened volume
    issue_created = [to_dt(it["created_at"]) for it in bundle["issues_opened"] if it.get("created_at")]
    df_issues_opened = series_by_day(issue_created, start, end)
    st.subheader("Issues opened (per day)")
    st.line_chart(df_issues_opened.set_index("day")["count"])

    # Metric 8: Issues closed volume and closure rate
    issue_closed = [to_dt(it["closed_at"]) for it in bundle["issues_closed"] if it.get("closed_at")]
    df_issues_closed = series_by_day(issue_closed, start, end)
    st.subheader("Issues closed (per day)")
    st.line_chart(df_issues_closed.set_index("day")["count"])

    opened_n = len(bundle["issues_opened"])
    closed_n = len(bundle["issues_closed"])
    closure_rate = (closed_n / opened_n) if opened_n > 0 else None
    c1, c2, c3 = st.columns(3)
    c1.metric("Issues opened (window)", f"{opened_n:,}")
    c2.metric("Issues closed (window)", f"{closed_n:,}")
    c3.metric("Closure rate (closed/opened)", "n/a" if closure_rate is None else f"{closure_rate:.2f}")

    st.divider()

    st.header("Popularity Metrics")

    # Metric 9: Stars (total and gained in window, best effort)
    stars_total = repo_info.get("stargazers_count", 0)
    stars_gained = len(bundle["stars_gained"])
    st.metric("Stars gained (window, best effort)", f"{stars_gained:,}")

    df_stars = series_by_day(bundle["stars_gained"], start, end) if bundle["stars_gained"] else None
    if df_stars is not None:
        st.subheader("Stars gained (per day, best effort)")
        st.line_chart(df_stars.set_index("day")["count"])
    else:
        st.info("Stars gained chart unavailable (no recent stargazer events fetched). Increase pages in code if needed.")

    # Metric 10: Forks (total and gained in window)
    forks_total = repo_info.get("forks_count", 0)
    forks_gained = len(bundle["forks_gained"])
    st.metric("Forks gained (window, best effort)", f"{forks_gained:,}")

    df_forks = series_by_day(bundle["forks_gained"], start, end) if bundle["forks_gained"] else None
    if df_forks is not None:
        st.subheader("Forks gained (per day, best effort)")
        st.line_chart(df_forks.set_index("day")["count"])
    else:
        st.info("Forks gained chart unavailable (no recent fork events fetched). Increase pages in code if needed.")

    st.divider()

    with st.expander("Notes on how metrics are computed"):
        st.write(
            """
- CI metrics use GitHub Actions workflow runs on the default branch within the selected window.
- Time to fix a broken build is approximated from failure streaks followed by a success.
- Active contributors are unique commit authors seen in the commits fetched for the window.
- Issues and PR counts come from the GitHub Search API and are subject to its 1000 result cap per query.
- Stars gained and forks gained are computed by fetching newest stargazers and forks until outside the window (best effort).
"""
        )


if __name__ == "__main__":
    main()
