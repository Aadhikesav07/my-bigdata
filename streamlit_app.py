import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import textwrap

DATA_PATH = "dataset/reddit_posts_clean.csv"

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    return df

df = load_data()

st.title("Reddit Submissions Analysis Dashboard")

# Flair distribution
if "link_flair_text" in df.columns:
    st.header("Distribution of Post Flair")
    flair_counts = df["link_flair_text"].value_counts().reset_index()
    flair_counts.columns = ["Flair", "Count"]
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(x="Flair", y="Count", data=flair_counts, palette="pastel", ax=ax)
    ax.set_xticklabels([textwrap.fill(l, 15) for l in flair_counts["Flair"]], rotation=15, ha="right")
    st.pyplot(fig)

# Avg upvotes by flair
if "link_flair_text" in df.columns and "ups" in df.columns:
    st.header("Average Upvotes by Post Flair")
    avg_ups = df.groupby("link_flair_text")["ups"].mean().sort_values(ascending=False).reset_index()
    avg_ups.columns = ["Flair", "Average Ups"]
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(x="Flair", y="Average Ups", data=avg_ups, palette="crest", ax=ax)
    ax.set_xticklabels([textwrap.fill(l, 12) for l in avg_ups["Flair"]], rotation=15, ha="right")
    st.pyplot(fig)

# Posts per weekday
if "weekday" in df.columns:
    st.header("Posts by Weekday")
    weekday_counts = df["weekday"].value_counts().reindex([
        "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"
    ]).dropna().reset_index()
    weekday_counts.columns = ["Weekday", "Count"]
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(x="Weekday", y="Count", data=weekday_counts, palette="mako", ax=ax)
    st.pyplot(fig)

# Upvote ratio stats
if "upvote_ratio" in df.columns:
    st.header("Upvote Ratio Statistics")
    st.write(f"**Maximum Upvote Ratio**: {df['upvote_ratio'].max():.2f}")
    st.write(f"**Average Upvote Ratio**: {df['upvote_ratio'].mean():.2f}")

# Filter by flair
if "link_flair_text" in df.columns:
    st.sidebar.header("Filter by Flair")
    flair_options = df["link_flair_text"].dropna().unique().tolist()
    selected_flair = st.sidebar.multiselect("Select Flair(s)", flair_options)
    if selected_flair:
        filtered = df[df["link_flair_text"].isin(selected_flair)]
        st.write(f"Showing {len(filtered)} posts with selected flairs.")
        st.dataframe(filtered.head(50))

with st.expander("Show raw data (first 100 rows)"):
    st.write(df.head(100))
    