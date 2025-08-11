"""
Parser for Terraforming Mars end-game HTML screens.

Extracts winner, generation count, and per-player breakdown from the HTML
rendered by the upstream web UI at /the-end?id=<player_id>.
"""
from typing import Dict, Any, List, Optional
from bs4 import BeautifulSoup


def _text(el) -> str:
    if el is None:
        return ""
    return " ".join(el.get_text(" ", strip=True).split())


def parse_end_screen(html: str) -> Dict[str, Any]:
    soup = BeautifulSoup(html, "html.parser")
    result: Dict[str, Any] = {
        "title": None,
        "winner": None,
        "generations": None,
        "players": [],
        "log_download_url": None,
    }

    # Title
    h1 = soup.select_one("#game-end h1")
    result["title"] = _text(h1) if h1 else None

    # Winner and generations
    winner_el = soup.select_one(".game-end-winer-announcement")
    if winner_el:
        # Expect format: <span><span class="log-player ...">NAME</span></span> <span>won!</span>
        name_span = winner_el.select_one(".log-player")
        if name_span:
            result["winner"] = _text(name_span)

    gens_h2 = soup.select_one(".game_end_victory_points h2")
    if gens_h2:
        # e.g., "Victory point breakdown after 20 generations"
        txt = _text(gens_h2)
        # Find integer in text
        import re
        m = re.search(r"after\s+(\d+)\s+generations?", txt)
        if m:
            result["generations"] = int(m.group(1))

    # Build header mapping from the table headers
    header_labels: List[str] = []
    header_ths = soup.select(".game_end_table thead th")
    for th in header_ths:
        label = ""
        # Detect by inner marker classes or text
        if th.select_one(".tr"):
            label = "TR"
        elif th.select_one(".m-and-a"):
            # Will appear twice: milestones then awards
            label = th.select_one(".m-and-a").get_text(strip=True)
        elif th.select_one(".table-forest-tile"):
            label = "FORESTS"
        elif th.select_one(".table-city-tile"):
            label = "CITIES"
        elif th.select_one(".table-moon-road-tile"):
            label = "MOON_ROAD"
        elif th.select_one(".table-moon-colony-tile"):
            label = "MOON_COLONY"
        elif th.select_one(".table-moon-mine-tile"):
            label = "MOON_MINE"
        elif th.select_one(".vp"):
            label = "VP"
        elif th.select_one(".game-end-total") or th.find("div", class_="game-end-total-column"):
            label = "TOTAL"
        elif th.select_one(".mc-icon"):
            label = "MC"
        elif _text(th):
            label = _text(th).upper()
        else:
            label = ""
        header_labels.append(label)

    # Parse rows
    rows = soup.select(".game_end_table tbody tr")
    for row in rows:
        tds = row.find_all("td")
        if not tds:
            continue
        # First cell: player link and corporation
        name_link = tds[0].select_one("a")
        corp_div = tds[0].select_one(".column-corporation")
        player_name = _text(name_link)
        corporation = _text(corp_div)

        # Remaining numeric cells map to headers (skip first header cell)
        stats: Dict[str, Any] = {
            "name": player_name,
            "corporation": corporation,
        }
        for idx, td in enumerate(tds[1:], start=1):
            label = header_labels[idx] if idx < len(header_labels) else f"COL_{idx}"
            value_text = _text(td)
            # Make numeric when possible
            try:
                value = int(value_text)
            except (ValueError, TypeError):
                value = value_text
            # Normalize duplicate labels like 'M'/'A'
            key_map = {
                "TR": "tr",
                "M": "milestones",
                "A": "awards",
                "FORESTS": "forests",
                "CITIES": "cities",
                "MOON_ROAD": "moon_road",
                "MOON_COLONY": "moon_colony",
                "MOON_MINE": "moon_mine",
                "VP": "vp",
                "TOTAL": "total",
                "MC": "mc",
            }
            key = key_map.get(label, label.lower() if label else f"col_{idx}")
            stats[key] = value

        result["players"].append(stats)

    # Log download link (if present)
    log_link = soup.select_one("a[href*='/api/game/logs']")
    if log_link and log_link.has_attr("href"):
        result["log_download_url"] = log_link["href"]

    return result


