import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import textwrap

# Page config
st.set_page_config(page_title="📱 Reddit Analysis", layout="wide", initial_sidebar_state="expanded")

# Custom CSS for better styling
st.markdown("""
<style>
    .main { padding-top: 2rem; }
    .stMetric { background-color: #f0f2f6; padding: 1rem; border-radius: 0.5rem; }
    h1 { color: #1f77b4; }
    h2 { color: #2c3e50; margin-top: 2rem; }
    .stPlotlyChart { border: 1px solid #e0e0e0; border-radius: 0.5rem; padding: 1rem; background: white; }
</style>
""", unsafe_allow_html=True)

# Data loading
DATA_PATH = "dataset/reddit_posts_clean.csv"

@st.cache_data(show_spinner=True)
def load_data():
    df = pd.read_csv(DATA_PATH)
    return df

df = load_data()

# Sidebar filters
st.sidebar.header("🎛️ Filter by Flair")
st.sidebar.markdown("Select Flair(s)")

if "link_flair_text" in df.columns:
    all_flairs = sorted(df["link_flair_text"].dropna().unique().tolist())
    selected_flairs = st.sidebar.multiselect(
        "Choose flairs to filter:",
        options=all_flairs,
        default=[],
        help="Select one or more flairs to filter the data"
    )
    
    # Apply filter
    if selected_flairs:
        df_filtered = df[df["link_flair_text"].isin(selected_flairs)].copy()
    else:
        df_filtered = df.copy()
else:
    df_filtered = df.copy()
    selected_flairs = []

# Display filter status
if selected_flairs:
    st.sidebar.success(f"✓ Filtering by: {', '.join(selected_flairs)}")
    st.sidebar.info(f"Showing {len(df_filtered):,} of {len(df):,} posts")
else:
    st.sidebar.info(f"Showing all {len(df):,} posts")

# Clear filters button
if selected_flairs and st.sidebar.button("🔄 Clear Filters"):
    st.experimental_rerun()

# Main header
st.title("📱 Reddit Submissions Analysis Dashboard")
st.markdown("---")

# KPI metrics
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("📊 Total Posts", f"{len(df_filtered):,}")

with col2:
    if "upvote_ratio" in df_filtered.columns:
        avg_ratio = df_filtered["upvote_ratio"].mean()
        st.metric("⬆️ Avg Upvote Ratio", f"{avg_ratio:.2f}")
    else:
        st.metric("⬆️ Avg Upvote Ratio", "N/A")

with col3:
    if "link_flair_text" in df_filtered.columns:
        unique_flairs = df_filtered["link_flair_text"].nunique()
        st.metric("🏷️ Unique Flairs", f"{unique_flairs}")
    else:
        st.metric("🏷️ Unique Flairs", "N/A")

st.markdown("---")

# Visualization functions
def wrap_labels(labels, width=15):
    """Wrap long labels for better display"""
    return [textwrap.fill(str(l), width=width) for l in labels]

# 1. Distribution of Post Flair
if "link_flair_text" in df_filtered.columns:
    st.header("📊 Distribution of Post Flair")
    
    flair_counts = df_filtered["link_flair_text"].value_counts().head(20).reset_index()
    flair_counts.columns = ["Flair", "Count"]
    
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    colors = sns.color_palette("husl", len(flair_counts))
    bars = ax1.bar(range(len(flair_counts)), flair_counts["Count"], color=colors)
    
    ax1.set_xlabel("Flair", fontsize=12, fontweight='bold')
    ax1.set_ylabel("Count", fontsize=12, fontweight='bold')
    ax1.set_title("Distribution of Post Flair", fontsize=14, fontweight='bold', pad=20)
    
    # Set x-ticks
    ax1.set_xticks(range(len(flair_counts)))
    ax1.set_xticklabels(wrap_labels(flair_counts["Flair"].tolist(), 15), rotation=45, ha="right")
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    st.pyplot(fig1)
    plt.close()

st.markdown("---")

# 2. Average Upvotes by Post Flair
if "link_flair_text" in df_filtered.columns and "ups" in df_filtered.columns:
    st.header("⬆️ Average Upvotes by Post Flair")
    
    avg_upvotes = df_filtered.groupby("link_flair_text")["ups"].mean().sort_values(ascending=False).head(20).reset_index()
    avg_upvotes.columns = ["Flair", "Average Ups"]
    
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    colors = sns.color_palette("viridis", len(avg_upvotes))
    bars = ax2.bar(range(len(avg_upvotes)), avg_upvotes["Average Ups"], color=colors)
    
    ax2.set_xlabel("Flair", fontsize=12, fontweight='bold')
    ax2.set_ylabel("Average Ups", fontsize=12, fontweight='bold')
    ax2.set_title("Average Upvotes by Post Flair", fontsize=14, fontweight='bold', pad=20)
    
    ax2.set_xticks(range(len(avg_upvotes)))
    ax2.set_xticklabels(wrap_labels(avg_upvotes["Flair"].tolist(), 15), rotation=45, ha="right")
    
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}',
                ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close()

st.markdown("---")

# 3. Posts by Weekday
if "weekday" in df_filtered.columns:
    st.header("📅 Posts by Weekday")
    
    # Define weekday order
    weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    
    weekday_counts = df_filtered["weekday"].value_counts().reindex(weekday_order, fill_value=0).reset_index()
    weekday_counts.columns = ["Weekday", "Count"]
    
    fig3, ax3 = plt.subplots(figsize=(12, 6))
    colors = sns.color_palette("coolwarm", len(weekday_counts))
    bars = ax3.bar(weekday_counts["Weekday"], weekday_counts["Count"], color=colors)
    
    ax3.set_xlabel("Weekday", fontsize=12, fontweight='bold')
    ax3.set_ylabel("Count", fontsize=12, fontweight='bold')
    ax3.set_title("Posts by Weekday", fontsize=14, fontweight='bold', pad=20)
    ax3.tick_params(axis='x', rotation=0)
    
    for bar in bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    st.pyplot(fig3)
    plt.close()

st.markdown("---")

# 4. Upvote Ratio Statistics
if "upvote_ratio" in df_filtered.columns:
    st.header("📈 Upvote Ratio Statistics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        max_ratio = df_filtered["upvote_ratio"].max()
        st.metric("Maximum Upvote Ratio", f"{max_ratio:.2f}")
    
    with col2:
        avg_ratio = df_filtered["upvote_ratio"].mean()
        st.metric("Average Upvote Ratio", f"{avg_ratio:.2f}")
    
    with col3:
        min_ratio = df_filtered["upvote_ratio"].min()
        st.metric("Minimum Upvote Ratio", f"{min_ratio:.2f}")

st.markdown("---")

# 5. Data Table with filters
st.header("📋 Filtered Data")

# Column selector
if st.checkbox("Show data table"):
    available_cols = df_filtered.columns.tolist()
    default_cols = [c for c in ["title", "link_flair_text", "ups", "upvote_ratio", "num_comments", "weekday"] if c in available_cols]
    
    selected_cols = st.multiselect(
        "Select columns to display:",
        options=available_cols,
        default=default_cols[:6] if default_cols else available_cols[:6]
    )
    
    if selected_cols:
        # Number of rows to display
        num_rows = st.slider("Number of rows to display:", 10, 100, 50, 10)
        st.dataframe(df_filtered[selected_cols].head(num_rows), use_container_width=True)
        
        # Download button
        csv = df_filtered[selected_cols].to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download filtered data as CSV",
            data=csv,
            file_name="reddit_filtered_data.csv",
            mime="text/csv"
        )

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #888; padding: 2rem;'>
    <p>📱 Reddit Submissions Analysis Dashboard | Built with Streamlit</p>
</div>
""", unsafe_allow_html=True)
