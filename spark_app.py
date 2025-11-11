import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import textwrap
import re

SRC = "dataset/data.csv"
OUT_DIR = "dataset"
os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(SRC)

def clean_text(text):
    if pd.isna(text):
        return ""
    text = re.sub(r'http\S+', '', str(text))
    text = re.sub(r'\[.*?\]\(.*?\)', '', text)
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

if 'selftext' in df.columns:
    df['clean_selftext'] = df['selftext'].apply(clean_text)
if 'title' in df.columns:
    df['clean_title'] = df['title'].apply(clean_text)

sns.set_theme(style="whitegrid", context="talk")

def save_fig(path, fig=None, dpi=180):
    if fig is None:
        fig = plt.gcf()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

def wrap_labels(labels, width=18):
    return [textwrap.fill(str(x), width=width) for x in labels]

# 1. Flair distribution - FIX: Added hue parameter
if "link_flair_text" in df.columns:
    flair_counts = df["link_flair_text"].value_counts().reset_index()
    flair_counts.columns = ["Flair", "Count"]
    
    fig, ax = plt.subplots(figsize=(7.5, 4.5), constrained_layout=True)
    sns.barplot(x="Flair", y="Count", hue="Flair", data=flair_counts, ax=ax, palette="pastel", legend=False)
    ax.set_xlabel("Flair")
    ax.set_ylabel("Number of posts")
    ax.set_title("Distribution of Post Flair", pad=10)
    
    # FIX: Set ticks before labels
    ax.set_xticks(range(len(flair_counts)))
    ax.set_xticklabels(wrap_labels(flair_counts["Flair"].tolist(), 12), rotation=15, ha="right")
    
    for c in ax.containers:
        ax.bar_label(c, fmt="%d", padding=3)
    
    save_fig(os.path.join(OUT_DIR, "flair_distribution.png"), fig)

# 2. Average upvotes by flair
if "link_flair_text" in df.columns and "ups" in df.columns:
    avg_upvotes = df.groupby("link_flair_text")["ups"].mean().sort_values(ascending=False).reset_index()
    avg_upvotes.columns = ["Flair", "Average Upvotes"]
    
    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    sns.barplot(x="Flair", y="Average Upvotes", hue="Flair", data=avg_upvotes, ax=ax, legend=False)
    ax.set_xlabel("Flair")
    ax.set_ylabel("Average Upvotes")
    ax.set_title("Average Upvotes by Post Flair", pad=10)
    
    # FIX: Set ticks before labels
    ax.set_xticks(range(len(avg_upvotes)))
    ax.set_xticklabels(wrap_labels(avg_upvotes["Flair"].tolist(), 12), rotation=15, ha="right")
    
    for c in ax.containers:
        ax.bar_label(c, fmt="%.1f", padding=3)
    
    save_fig(os.path.join(OUT_DIR, "flair_avg_ups.png"), fig)

# 3. Posts by weekday - FIX: Added hue parameter
if "weekday" in df.columns:
    weekday_counts = df["weekday"].value_counts().reset_index()
    weekday_counts.columns = ["Weekday", "Count"]
    
    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    sns.barplot(x="Weekday", y="Count", hue="Weekday", data=weekday_counts, ax=ax, palette="mako", legend=False)
    ax.set_xlabel("Weekday")
    ax.set_ylabel("Number of posts")
    ax.set_title("Posts by Weekday", pad=10)
    
    for c in ax.containers:
        ax.bar_label(c, fmt="%d", padding=3)
    
    save_fig(os.path.join(OUT_DIR, "posts_by_weekday.png"), fig)

# 4. Upvote ratio statistics
if "upvote_ratio" in df.columns:
    print(f"\nUpvote Ratio Statistics:")
    print(f"Maximum: {df['upvote_ratio'].max():.2f}")
    print(f"Average: {df['upvote_ratio'].mean():.2f}")
    print(f"Minimum: {df['upvote_ratio'].min():.2f}")

# 5. Export cleaned dataset
clean_cols = [c for c in df.columns if c in {"selftext", "title", "clean_selftext", "clean_title", 
                                               "link_flair_text", "ups", "upvote_ratio", 
                                               "num_comments", "weekday", "author", "created"}]
df[clean_cols].to_csv(os.path.join(OUT_DIR, "reddit_posts_clean.csv"), index=False)

print("\nAll analyses and visualizations saved in 'dataset/' folder.")
print("Check PNG files and reddit_posts_clean.csv for details.")
