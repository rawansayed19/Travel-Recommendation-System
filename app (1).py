# Smart Travel Destination Recommendation System
# Run locally:  streamlit run app.py
# Needs travel_dataset.csv in the same folder as this file.

from pathlib import Path

import pandas as pd
import streamlit as st

DATA_PATH = Path(__file__).parent / "travel_dataset.csv"

# ---------------------------------------------------------------
# 1) Settings (all values the UI can send to the model)
# ---------------------------------------------------------------
BUDGET_OPTIONS = ["Low", "Medium", "High"]
DURATION_OPTIONS = {"3 days": 3, "1 week": 7, "2 weeks": 14, "1 month": 30}
WEATHER_OPTIONS = ["Hot", "Moderate", "Cold"]
ACTIVITY_OPTIONS = ["Beach", "Adventure", "History", "Nightlife",
                    "Nature", "Shopping", "Hiking"]

BUDGET_LEVEL = {"low": 1, "medium": 2, "high": 3}
# Max total trip cost (USD) accepted for each budget level
TRIP_CAPS = {"low": 800, "medium": 2500, "high": 6000}
# Typical daily cost per budget level (used only as a tie-breaker)
TYPICAL_COST = {"low": 70, "medium": 175, "high": 425}
# Dataset climate -> weather option shown in the UI
WEATHER_MAP = {"Tropical": "hot", "Desert": "hot", "Temperate": "moderate",
               "Mediterranean": "moderate", "Cold": "cold"}
# UI activity -> feature column (vibe column, or a word searched in the activities text)
ACT_COLS = {"beach": "a_beach", "adventure": "a_adventure", "history": "a_historic",
            "nightlife": "a_nightlife", "nature": "a_nature",
            "shopping": "a_shopping", "hiking": "a_hiking"}
# Weights from the project report
WEIGHTS = {"budget": 0.30, "activities": 0.30, "weather": 0.20, "duration": 0.20}

# One fixed, realistic climate per city (climate is a property of the city)
CITY_CLIMATE = {
    "Amsterdam, Netherlands": "Temperate", "Athens, Greece": "Mediterranean",
    "Auckland, New Zealand": "Temperate", "Bali, Indonesia": "Tropical",
    "Bangkok, Thailand": "Tropical", "Barcelona, Spain": "Mediterranean",
    "Budapest, Hungary": "Temperate", "Buenos Aires, Argentina": "Temperate",
    "Cairo, Egypt": "Desert", "Cape Town, South Africa": "Mediterranean",
    "Cartagena, Colombia": "Tropical", "Chefchaouen, Morocco": "Mediterranean",
    "Chicago, USA": "Cold", "Copenhagen, Denmark": "Temperate",
    "Cusco, Peru": "Temperate", "Dubrovnik, Croatia": "Mediterranean",
    "Edinburgh, Scotland": "Temperate", "Fiji, Fiji": "Tropical",
    "Gizeh, Egypt": "Desert", "Goa, India": "Tropical",
    "Hobart, Australia": "Temperate", "Hoi An, Vietnam": "Tropical",
    "Kathmandu, Nepal": "Temperate", "Kigali, Rwanda": "Temperate",
    "Kyoto, Japan": "Temperate", "Lima, Peru": "Temperate",
    "Lisbon, Portugal": "Mediterranean", "Los Angeles, USA": "Mediterranean",
    "Luang Prabang, Laos": "Tropical", "Malé, Maldives": "Tropical",
    "Marrakesh, Morocco": "Desert", "Medellín, Colombia": "Temperate",
    "Melbourne, Australia": "Temperate", "Mexico City, Mexico": "Temperate",
    "Miami, USA": "Tropical", "Montevideo, Uruguay": "Temperate",
    "Montreal, Canada": "Cold", "Nairobi, Kenya": "Temperate",
    "New York, USA": "Temperate", "Paris, France": "Temperate",
    "Perth, Australia": "Mediterranean", "Prague, Czechia": "Temperate",
    "Quebec City, Canada": "Cold", "Queenstown, New Zealand": "Temperate",
    "Quito, Ecuador": "Temperate", "Reykjavik, Iceland": "Cold",
    "Rio de Janeiro, Brazil": "Tropical", "Rome, Italy": "Mediterranean",
    "San Francisco, USA": "Mediterranean", "Santiago, Chile": "Mediterranean",
    "Seoul, South Korea": "Temperate", "Seychelles, Seychelles": "Tropical",
    "Siem Reap, Cambodia": "Tropical", "Singapore, Singapore": "Tropical",
    "Sydney, Australia": "Temperate", "Tahiti, French Polynesia": "Tropical",
    "Taipei, Taiwan": "Tropical", "Tulum, Mexico": "Tropical",
    "Vancouver, Canada": "Temperate", "Zanzibar City, Tanzania": "Tropical",
}


# ---------------------------------------------------------------
# 2) Data loading and preparation (same steps as the notebook)
# ---------------------------------------------------------------
@st.cache_data
def load_data():
    data = pd.read_csv(DATA_PATH)
    for col in data.select_dtypes(include=["object", "string"]).columns:
        data[col] = data[col].str.strip()
    data["budget_level"] = data["budget"].map({"$": 1, "$$": 2, "$$$": 3})
    data["climate"] = data["name"].map(CITY_CLIMATE).fillna(data["climate"])
    data["weather"] = data["climate"].map(WEATHER_MAP)

    # Activity features: the vibe column + words searched in the activities text
    acts = pd.get_dummies(data["vibe"].str.lower(), prefix="a").astype(float)
    text = data["activities"].str.lower()
    acts["a_shopping"] = text.str.contains("shopping").astype(float)
    acts["a_hiking"] = text.str.contains("hiking").astype(float)
    for col in ACT_COLS.values():          # make sure every column exists
        if col not in acts:
            acts[col] = 0.0
    return data, acts


df, acts = load_data()


# ---------------------------------------------------------------
# 3) Recommendation model (Weighted Scoring)
# ---------------------------------------------------------------
def recommend(budget, duration, activities, weather, top_n=5):
    """Return the top_n destinations as a DataFrame with a match percentage."""
    bl = budget.lower()
    days = DURATION_OPTIONS[duration]
    result = df.copy()

    # Budget: 1 if same level, 0.5 if one level away, otherwise 0
    diff = (result["budget_level"] - BUDGET_LEVEL[bl]).abs()
    result["budget_s"] = (1 - diff * 0.5).clip(lower=0)

    # Duration: total trip cost must stay under the cap of the chosen budget
    total_cost = result["typical_cost_per_day_usd"] * days
    result["duration_s"] = (TRIP_CAPS[bl] / total_cost).clip(upper=1)

    # Weather: exact match with the chosen weather
    result["weather_s"] = (result["weather"] == weather.lower()).astype(float)

    # Activities: 1 if any chosen activity matches this destination
    hit = pd.Series(False, index=result.index)
    for act in activities:
        hit |= acts[ACT_COLS[act.lower()]] == 1
    result["act_s"] = hit.astype(float)

    result["match_pct"] = (100 * (
        WEIGHTS["budget"] * result["budget_s"]
        + WEIGHTS["activities"] * result["act_s"]
        + WEIGHTS["weather"] * result["weather_s"]
        + WEIGHTS["duration"] * result["duration_s"])).round(1)

    # Tie-breaker: daily cost close to the typical cost of the chosen budget
    closeness = (1 - (result["typical_cost_per_day_usd"] - TYPICAL_COST[bl]).abs()
                 / TYPICAL_COST[bl]).clip(lower=0)
    result["rank_score"] = result["match_pct"] + 0.1 * closeness

    # The dataset has several rows per destination: keep the best one
    result = result.sort_values("rank_score", ascending=False).drop_duplicates("name")
    result["days"] = days
    result["trip_cost"] = result["typical_cost_per_day_usd"] * days
    result["matched"] = [
        [x for x in activities if acts.loc[i, ACT_COLS[x.lower()]] == 1]
        for i in result.index
    ]
    return result.head(top_n)


# ---------------------------------------------------------------
# 4) User interface
# ---------------------------------------------------------------
st.set_page_config(page_title="Smart Travel Recommender", page_icon="🌍")

st.title("🌍 Smart Travel Destination Recommender")
st.write("Tell us about your trip and we will suggest the destinations that fit you best.")

# Daily cost range of each budget level, taken from the dataset itself
ranges = df.groupby("budget_level")["typical_cost_per_day_usd"].agg(["min", "max"])
budget_help = " | ".join(
    f"{name}: ${int(ranges.loc[lvl, 'min'])}-{int(ranges.loc[lvl, 'max'])} per day"
    for name, lvl in zip(["Low", "Medium", "High"], [1, 2, 3])
)

with st.form("preferences"):
    c1, c2 = st.columns(2)
    budget = c1.selectbox("Budget", BUDGET_OPTIONS, index=1, help=budget_help)
    duration = c2.selectbox("Trip duration", list(DURATION_OPTIONS), index=1)
    weather = c1.selectbox("Preferred weather", WEATHER_OPTIONS, index=1,
                           help="Hot: tropical and desert. Moderate: temperate and "
                                "Mediterranean. Cold: cold climates.")
    activities = c2.multiselect("What do you love to do? (one or more)",
                                ACTIVITY_OPTIONS, default=["History"])
    submitted = st.form_submit_button("Find my destinations", type="primary")

if submitted:
    if not activities:
        st.warning("Please choose at least one activity.")
    else:
        results = recommend(budget, duration, activities, weather)
        st.subheader("Top recommendations")
        st.caption(f"{budget} budget | {duration} | {weather} weather | "
                   f"{', '.join(activities)}")

        for rank, (_, r) in enumerate(results.iterrows(), 1):
            stars = round(r["match_pct"] / 20)
            with st.container(border=True):
                left, right = st.columns([3, 1])
                left.markdown(f"### {rank}. {r['name']}")
                parts = []
                if r["matched"]:
                    parts.append("Matches: " + ", ".join(r["matched"]))
                parts.append(f"{r['climate']} climate")
                left.caption(" | ".join(parts))
                right.markdown(f"**{r['match_pct']:.0f}% match**")
                right.markdown("★" * stars + "☆" * (5 - stars))
                st.progress(min(r["match_pct"] / 100, 1.0))
                st.markdown(
                    f"**Estimated trip cost:** about ${int(r['trip_cost']):,} "
                    f"for {int(r['days'])} days (${int(r['typical_cost_per_day_usd'])}/day)"
                )
                with st.expander("Details"):
                    st.write(r["description"])
                    st.write(f"**Trip style:** {r['vibe']}  |  **Best for:** {r['group_size']}")
                    st.write(f"**Best months:** {r['best_months']}")
                    st.write(f"**Things to do:** {r['activities']}")

with st.expander("How the recommendation score works"):
    st.write(
        "Every destination gets a score out of 100: budget fit (30%), matching "
        "activities (30%), weather (20%) and trip cost for your duration (20%). "
        "Destinations with the same score are ordered by how close their daily "
        "cost is to the typical cost of your budget."
    )
