# AI Powered Content Creation and Personalization 
This project uses AI to generate personalized social media posts based on user interests. It leverages the GPT-2 model from Hugging Face's Transformers library to create engaging content tailored to individual preferences.

## Features
- Generates personalized posts for users based on their likes (e.g., tech, fitness, travel).
- Supports multiple interest categories with varied tones.
- Cleans up AI-generated text to ensure neat, readable output.
- Saves results to a CSV file for easy access.

## Requirements
- Python 3.6+
- PyTorch
- Transformers (Hugging Face)
- Pandas
- CUDA-compatible GPU (optional, for faster processing)

## Installation

1. Clone this repository or download the script
2. Install the required packages:

```bash
pip install pandas torch transformers
```

## Usage

Simply run the script:

```bash
python personalize_posts.py
```

The script will:
1. Load the GPT-2 model
2. Process mock user data (or replace with your own data)
3. Generate personalized posts
4. Display results in the console
5. Save the posts to a CSV file (`personalized_posts.csv`)

## Output Example
The script will generate a table showing:
- User ID
- Interest category
- Generated post
- Engagement metrics (likes received)

## Customization
You can customize the script by:
- Replacing the mock user data with real user data
- Modifying the templates in the `templates` dictionary
- Adjusting the interest categorization logic in `assign_interest()`
- Changing the topic selection logic in `generate_post()`

## Performance Notes
- The script will automatically use GPU acceleration if available
- Processing time depends on the number of users and your hardware


## License
- This project is unlicensed; use it freely for educational or personal purposes. For commercial use, don't hesitate to get in touch with the author.

## Contributing
- You can fix this project, submit pull requests, or open issues for bugs and feature requests.
