import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
from textblob import TextBlob
from sklearn.feature_extraction.text import CountVectorizer
import seaborn as sns
import numpy as np

st.set_page_config(page_title="📱 Reddit Analysis", layout="wide")

# Data loading
DATA_PATH = "dataset/reddit_posts_clean.csv"

@st.cache_data(show_spinner=True)
def load_data():
    df = pd.read_csv(DATA_PATH)
    return df

df = load_data()

def apply_filters(df):
    st.sidebar.header("🎛️ Filter by Flair")
    flairs = sorted(df["link_flair_text"].dropna().unique()) if "link_flair_text" in df else []
    filtered = df
    selected_flairs = []
    if flairs:
        selected_flairs = st.sidebar.multiselect(
            "Choose flairs to filter:", 
            options=flairs,
            help="Select one or more flairs to filter the data",
            default=[]
        )
        if selected_flairs:
            filtered = filtered[filtered["link_flair_text"].isin(selected_flairs)]
    st.sidebar.info(f"Showing {len(filtered):,} of {len(df):,} posts")
    if selected_flairs and st.sidebar.button("🔄 Clear Filters"):
        st.session_state["clear"] = True
    return filtered

if "clear" in st.session_state:
    st.session_state.pop("clear")
    st.experimental_rerun()

df_filtered = apply_filters(df)

# -------------------------------------------
# KPI Metrics
# -------------------------------------------
cols = st.columns(4)
with cols[0]:
    st.metric("Total Posts", f"{len(df_filtered):,}")
with cols[1]:
    st.metric("Avg Upvote Ratio", 
              f"{df_filtered['upvote_ratio'].mean():.2f}" if "upvote_ratio" in df_filtered else "N/A")
with cols[2]:
    st.metric("Unique Flairs", 
              df_filtered["link_flair_text"].nunique() if "link_flair_text" in df_filtered else "N/A")
with cols[3]:
    st.metric("Max Upvotes", 
              int(df_filtered["ups"].max()) if "ups" in df_filtered else "N/A")

st.markdown("---")

# -------------------------------------------
# Flair Distribution Pie Chart
# -------------------------------------------
if "link_flair_text" in df_filtered:
    flair_counts = df_filtered["link_flair_text"].value_counts().reset_index()
    flair_counts.columns = ["Flair", "Count"]
    fig = px.pie(flair_counts, values="Count", names="Flair", title="Flair Distribution")
    fig.update_traces(textinfo='value+label')
    st.plotly_chart(fig, use_container_width=True)

# -------------------------------------------
# Most Common Words (NLP)
# -------------------------------------------
st.subheader("📝 Most Common Words in Titles (NLP)")

def get_top_words(texts, n=15):
    cv = CountVectorizer(stop_words="english", max_features=n)
    counts = cv.fit_transform(texts.dropna())
    word_freq = dict(zip(cv.get_feature_names_out(), counts.sum(axis=0).A1))
    return word_freq

words = get_top_words(df_filtered['title'] if 'title' in df_filtered else pd.Series([]), 15)
word_freq_df = pd.DataFrame.from_dict(words, orient='index').reset_index()
word_freq_df.columns = ["Word", "Frequency"]
fig = px.bar(word_freq_df, x="Word", y="Frequency", title="Top Words in Titles", color="Frequency", color_continuous_scale="viridis")
st.plotly_chart(fig, use_container_width=True)

# -------------------------------------------
# Word Cloud
# -------------------------------------------
st.subheader("☁️ Word Cloud of Post Titles")
text = " ".join(t for t in df_filtered['title'].dropna().astype(str))
stopwords = set(STOPWORDS)
wordcloud = WordCloud(stopwords=stopwords, background_color="white", width=800, height=300, colormap='plasma').generate(text)
fig, ax = plt.subplots(figsize=(10, 3))
ax.imshow(wordcloud, interpolation="bilinear")
ax.axis("off")
st.pyplot(fig)
plt.close(fig)

# -------------------------------------------
# Sentiment Analysis
# -------------------------------------------
st.subheader("😊 Sentiment Analysis (TextBlob Polarity)")

def calc_sentiment(txt):
    try:
        return TextBlob(txt).sentiment.polarity
    except Exception:
        return 0.0

if 'title' in df_filtered:
    df_filtered['sentiment'] = df_filtered['title'].fillna("").apply(calc_sentiment)
    sentiment_fig = px.histogram(df_filtered, x="sentiment", nbins=25,
                                 title="Sentiment Polarity of Titles",
                                 color_discrete_sequence=["teal"])
    st.plotly_chart(sentiment_fig, use_container_width=True)

    pol_mean = df_filtered["sentiment"].mean()
    if pol_mean > 0.1:
        st.success(f"Overall Positive Sentiment (avg polarity: **{pol_mean:.2f}**)")
    elif pol_mean < -0.1:
        st.error(f"Overall Negative Sentiment (avg polarity: **{pol_mean:.2f}**)")
    else:
        st.info(f"Overall Neutral Sentiment (avg polarity: **{pol_mean:.2f}**)")

# -------------------------------------------
# Comment Count Distribution
# -------------------------------------------
if "num_comments" in df_filtered:
    st.subheader("💬 Comment Count Distribution")
    fig = px.histogram(df_filtered, x="num_comments", nbins=25, color_discrete_sequence=["#FF7F0E"])
    st.plotly_chart(fig, use_container_width=True)

# -------------------------------------------
# Upvote Analysis by Weekday
# -------------------------------------------
if "weekday" in df_filtered and "ups" in df_filtered:
    st.subheader("📅 Upvotes by Weekday")
    weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    temp_df = df_filtered.groupby("weekday")["ups"].mean().reindex(weekday_order).reset_index()
    fig = px.bar(temp_df, x="weekday", y="ups", title="Average Upvotes per Weekday", color="ups", color_continuous_scale="blues")
    st.plotly_chart(fig, use_container_width=True)

# -------------------------------------------
# Data Table with Filters and Download
# -------------------------------------------
st.header("📋 Data Preview")
if st.checkbox("Show data table (with filter)"):
    tbl_cols = st.multiselect(
        "Columns to display:",
        df_filtered.columns.tolist(),
        default=["title", "link_flair_text", "ups", "upvote_ratio", "num_comments", "weekday"] if "link_flair_text" in df_filtered else df_filtered.columns.tolist()[:6]
    )
    tbl_len = st.slider("Rows to show:", 10, 100, 30, step=10)
    st.dataframe(df_filtered[tbl_cols].head(tbl_len), use_container_width=True)
    csv = df_filtered[tbl_cols].to_csv(index=False).encode('utf-8')
    st.download_button("⬇️ Download CSV", csv, "filtered_reddit_data.csv", "text/csv")

# Footer
st.markdown("---")
st.markdown("""
<p style='text-align: center; color: var(--text-color); opacity: 0.7;'>
    📱 <b>Reddit Submissions Analysis Dashboard | Built with Streamlit, Plotly, and NLP</b>
</p>
""", unsafe_allow_html=True)
