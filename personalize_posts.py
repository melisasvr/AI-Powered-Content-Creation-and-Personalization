import pandas as pd
import torch
from transformers import pipeline, set_seed
import logging
import warnings
import re

# Suppress unnecessary warnings and logs
warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger("transformers").setLevel(logging.ERROR)

# Set a seed for reproducibility
set_seed(42)

# Check if PyTorch is available and inform user
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on {device.upper()}.")

# Initialize the text generation model (GPT-2) with PyTorch
try:
    generator = pipeline("text-generation", model="gpt2", max_new_tokens=10, framework="pt", device=0 if device == "cuda" else -1)
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

# Mock user data for 20 users
data = {
    "user_id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
    "likes": [
        ["tech", "gadgets"], ["fitness", "yoga"], ["travel", "food"], ["tech", "gaming"],
        ["fitness", "running"], ["travel", "photography"], ["tech", "AI"], ["fitness", "weights"],
        ["travel", "culture"], ["food", "movies"], ["tech", "coding"], ["fitness", "cycling"],
        ["travel", "hiking"], ["gaming", "music"], ["fitness", "swimming"], ["travel", "art"],
        ["tech", "robots"], ["fitness", "dance"], ["travel", "nature"], ["food", "books"]
    ],
    "post_history": [
        "Read about AI!", "Did a 5K run", "Visited Paris", "Beat a game level",
        "Set a new PR", "Snapped a sunset", "Coded a bot", "Lifted heavy",
        "Explored a museum", "Watched a thriller", "Wrote some code", "Biked a trail",
        "Hiked a mountain", "Played a new song", "Swam laps", "Saw an exhibit",
        "Built a robot", "Danced all night", "Camped in the wild", "Read a novel"
    ]
}
users_df = pd.DataFrame(data)

# User profiling function
def assign_interest(likes):
    interests = []
    if "tech" in likes or "gadgets" in likes or "gaming" in likes or "AI" in likes or "coding" in likes or "robots" in likes:
        interests.append("tech-savvy")
    if "fitness" in likes or "yoga" in likes or "running" in likes or "weights" in likes or "cycling" in likes or "swimming" in likes or "dance" in likes:
        interests.append("fitness-enthusiast")
    if "travel" in likes or "photography" in likes or "culture" in likes or "hiking" in likes or "art" in likes or "nature" in likes:
        interests.append("travel-lover")
    if "food" in likes or "movies" in likes or "music" in likes or "books" in likes and not interests:
        interests.append("general")
    return interests[0] if interests else "general"

# Assign interests to users
users_df["interest"] = users_df["likes"].apply(assign_interest)

# Advanced templates with tone variations
templates = {
    "tech-savvy": [
        "Hey {user_id}, {topic} is taking over! #TechLife",
        "{user_id}, geek out on this {topic} update! #Innovation"
    ],
    "fitness-enthusiast": [
        "Yo {user_id}, smash your {topic} routine! #FitFam",
        "{user_id}, level up with {topic} today! #StayStrong"
    ],
    "travel-lover": [
        "Hey {user_id}, {topic} vibes calling you! #Wanderlust",
        "{user_id}, pack your bags for {topic}! #AdventureAwaits"
    ],
    "general": [
        "Hi {user_id}, here’s a {topic} nugget! #StayCurious",
        "{user_id}, enjoy this {topic} tidbit! #GoodVibes"
    ]
}

# Content generation function with neat cleanup
def generate_post(user_id, interest, tone_index=0):
    topic = (
        "AI" if interest == "tech-savvy" and "AI" in users_df[users_df["user_id"] == user_id]["likes"].values[0]
        else "gaming" if interest == "tech-savvy" and "gaming" in users_df[users_df["user_id"] == user_id]["likes"].values[0]
        else "coding" if interest == "tech-savvy" and "coding" in users_df[users_df["user_id"] == user_id]["likes"].values[0]
        else "robots" if interest == "tech-savvy" and "robots" in users_df[users_df["user_id"] == user_id]["likes"].values[0]
        else "tech" if interest == "tech-savvy"
        else "workouts" if interest == "fitness-enthusiast" and "weights" in users_df[users_df["user_id"] == user_id]["likes"].values[0]
        else "running" if interest == "fitness-enthusiast" and "running" in users_df[users_df["user_id"] == user_id]["likes"].values[0]
        else "yoga" if interest == "fitness-enthusiast" and "yoga" in users_df[users_df["user_id"] == user_id]["likes"].values[0]
        else "cycling" if interest == "fitness-enthusiast" and "cycling" in users_df[users_df["user_id"] == user_id]["likes"].values[0]
        else "swimming" if interest == "fitness-enthusiast" and "swimming" in users_df[users_df["user_id"] == user_id]["likes"].values[0]
        else "dance" if interest == "fitness-enthusiast"
        else "new destinations" if interest == "travel-lover" and "travel" in users_df[users_df["user_id"] == user_id]["likes"].values[0]
        else "photography" if interest == "travel-lover" and "photography" in users_df[users_df["user_id"] == user_id]["likes"].values[0]
        else "culture" if interest == "travel-lover" and "culture" in users_df[users_df["user_id"] == user_id]["likes"].values[0]
        else "hiking" if interest == "travel-lover" and "hiking" in users_df[users_df["user_id"] == user_id]["likes"].values[0]
        else "art" if interest == "travel-lover" and "art" in users_df[users_df["user_id"] == user_id]["likes"].values[0]
        else "nature" if interest == "travel-lover"
        else "movies" if "movies" in users_df[users_df["user_id"] == user_id]["likes"].values[0]
        else "music" if "music" in users_df[users_df["user_id"] == user_id]["likes"].values[0]
        else "books"
    )
    prompt = templates[interest][tone_index % len(templates[interest])].format(user_id=str(user_id), topic=topic)
    try:
        generated = generator(prompt, clean_up_tokenization_spaces=True)[0]["generated_text"]
        # Neat cleanup: protect user_id, keep only intended hashtag, remove stray text
        allowed_hashtags = ["#TechLife", "#Innovation", "#FitFam", "#StayStrong", "#Wanderlust", "#AdventureAwaits", "#StayCurious", "#GoodVibes"]
        temp_placeholder = f"USERID{user_id}TEMP"
        cleaned = generated.replace(str(user_id), temp_placeholder)
        cleaned = re.sub(r"#\w+", lambda m: m.group(0) if m.group(0) in allowed_hashtags else "", cleaned)  # Keep only allowed hashtags
        cleaned = re.sub(r"\b\d+\b|http[s]?://\S+|www\.\S+|[-:;@–\"']|\s{2,}", "", cleaned).strip()  # Remove numbers, URLs, punctuation, quotes, extra spaces
        cleaned = cleaned.replace(temp_placeholder, str(user_id))  # Restore user_id
        # Trim to a consistent length (e.g., 12 words)
        words = cleaned.split()
        return " ".join(words[:12]) if len(words) >= 12 else " ".join(words + [""] * (12 - len(words)))  # Pad short posts with spaces
    except Exception as e:
        print(f"Error generating post for user {user_id}: {e}")
        return prompt

# Generate posts for all users (alternate tones for variety)
users_df["post"] = [generate_post(row["user_id"], row["interest"], i % 2) for i, row in users_df.iterrows()]

# Simulate feedback (mock likes received)
users_df["likes_received"] = [10, 25, 15, 30, 20, 18, 22, 28, 12, 17, 14, 19, 23, 16, 21, 13, 27, 11, 24, 18]

# Display results in a clean, formatted table
print("\nGenerated Personalized Posts:")
pd.set_option("display.max_colwidth", 70)  # Limit column width for neatness
pd.set_option("display.width", 1000)
print(users_df[["user_id", "interest", "post", "likes_received"]].to_string(index=False, col_space={"user_id": 8, "interest": 20, "post": 50, "likes_received": 12}))

# Save to a file
users_df[["user_id", "post"]].to_csv("personalized_posts.csv", index=False)
print("\nPosts saved to 'personalized_posts.csv'")