import sys
import ast
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
from fuzzywuzzy import process
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize, MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import hstack, csr_matrix, vstack

def parse_watchlist(xml_file):
    """
    Parses the user's anime watchlist from a MyAnimeList XML export.
    Only includes anime with a watched episodes count greater than 0.
    Returns a list of anime titles.
    """
    tree = ET.parse(xml_file)
    root = tree.getroot()
    watchlist = []
    for anime in root.findall('anime'):
        watched_elem = anime.find('my_watched_episodes')
        try:
            watched_count = int(watched_elem.text.strip()) if watched_elem is not None and watched_elem.text else 0
        except ValueError:
            watched_count = 0
        # Only include anime if watched episodes > 0
        if watched_count > 0:
            title_elem = anime.find('series_title')
            if title_elem is not None and title_elem.text:
                watchlist.append(title_elem.text.strip())
    return watchlist

def load_anime_dataset(csv_file):
    """
    Loads the Kaggle anime dataset from a CSV file.
    Fills missing values for required columns.
    """
    df = pd.read_csv(csv_file)
    required_columns = ['genres', 'averageScore', 'popularity', 'description', 'tags',
                        'title_romaji', 'title_english', 'title_native']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
        # For numeric columns, use 0; otherwise, empty string.
        if col in ['averageScore', 'popularity']:
            df[col] = df[col].fillna(0)
        else:
            df[col] = df[col].fillna('')
    return df

def match_watchlist_titles(watchlist, df, threshold=70):
    """
    Uses fuzzy matching to align watchlist titles with the dataset.
    Checks against title_romaji, title_english, and title_native fields.
    Returns a dict mapping watchlist titles to the matched dataset index.
    If no match is found above the threshold, a warning is printed.
    """
    matched_indices = {}
    for title in watchlist:
        best_score = -1
        best_idx = None
        for idx, row in df.iterrows():
            candidate_titles = [str(row['title_romaji']), str(row['title_english']), str(row['title_native'])]
            for candidate in candidate_titles:
                if candidate.strip() == '':
                    continue
                _, score = process.extractOne(title, [candidate])
                if score > best_score:
                    best_score = score
                    best_idx = idx
        if best_score < threshold:
            print(f"Warning: No good match found for title '{title}'.")
        else:
            matched_indices[title] = best_idx
    return matched_indices

def process_genres(df):
    """
    Processes the 'genres' column.
    Converts string representation of lists into Python lists,
    then one-hot encodes using MultiLabelBinarizer.
    """
    from sklearn.preprocessing import MultiLabelBinarizer
    genres_list = []
    for genres_str in df['genres']:
        try:
            genres = ast.literal_eval(genres_str)
            if not isinstance(genres, list):
                genres = []
        except Exception:
            genres = []
        genres_list.append(genres)
    mlb = MultiLabelBinarizer()
    genres_encoded = mlb.fit_transform(genres_list)
    return genres_encoded, mlb

def process_tags(df):
    """
    Processes the 'tags' column.
    Parses the column (which is a string representing a list of dictionaries),
    extracts tag names and converts their ranks into a percentage (rank/100),
    then encodes them into a vector for each anime based on all tags found in the dataset.
    """
    tags_dicts = []
    for tags_str in df['tags']:
        try:
            tags_data = ast.literal_eval(tags_str)
        except Exception:
            tags_data = []
        tag_dict = {}
        if isinstance(tags_data, list):
            for tag in tags_data:
                if isinstance(tag, dict) and 'name' in tag and 'rank' in tag:
                    try:
                        tag_percentage = float(tag['rank']) / 100.0
                    except Exception:
                        tag_percentage = 0
                    tag_dict[tag['name']] = tag_percentage
        tags_dicts.append(tag_dict)
    
    # Build a set of all tag names across the dataset
    all_tags = set()
    for d in tags_dicts:
        for tag in d.keys():
            all_tags.add(tag)
    all_tags = sorted(list(all_tags))  # Sorted for consistency
    
    # Encode each anime's tags into a vector with weighted percentages
    tags_vectors = []
    for d in tags_dicts:
        vector = [d.get(tag, 0) for tag in all_tags]
        tags_vectors.append(vector)
    
    tags_encoded = np.array(tags_vectors)
    return tags_encoded, all_tags

def process_description(df):
    """
    Processes the anime description using TF-IDF vectorization.
    Returns the normalized TF-IDF matrix.
    """
    descriptions = df['description'].fillna('').tolist()
    tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
    tfidf_matrix = tfidf.fit_transform(descriptions)
    tfidf_matrix = normalize(tfidf_matrix)
    return tfidf_matrix, tfidf

def process_numeric_feature(df, column_name):
    """
    Processes a numeric column (e.g., averageScore or popularity)
    using MinMax scaling. Replaces empty strings with 0.
    Returns the scaled column.
    """
    scaler = MinMaxScaler()
    # Replace empty strings with 0 before converting to float
    values = df[[column_name]].replace('', 0).astype(float).fillna(0)
    scaled = scaler.fit_transform(values)
    return scaled, scaler

def combine_features(tfidf_matrix, genres_encoded, tags_encoded, score_scaled, popularity_scaled,
                     weights={'synopsis': 0.4, 'genres': 0.2, 'tags': 0.2, 'score': 0.1, 'popularity': 0.3}):
    """
    Combines the different features into a single vector for each anime.
    Each feature is first normalized and weighted.
    The resulting vectors are horizontally stacked and then normalized.
    """
    genres_norm = normalize(genres_encoded, norm='l2', axis=1)
    tags_norm = normalize(tags_encoded, norm='l2', axis=1) if tags_encoded.shape[1] > 0 else tags_encoded
    score_norm = normalize(score_scaled, norm='l2', axis=0)
    popularity_norm = normalize(popularity_scaled, norm='l2', axis=0)
    
    synopsis_weighted = tfidf_matrix * weights['synopsis']
    genres_weighted = genres_norm * weights['genres']
    tags_weighted = tags_norm * weights['tags']
    score_weighted = score_norm * weights['score']
    popularity_weighted = popularity_norm * weights['popularity']
    
    genres_sparse = csr_matrix(genres_weighted)
    tags_sparse = csr_matrix(tags_weighted)
    score_sparse = csr_matrix(score_weighted)
    popularity_sparse = csr_matrix(popularity_weighted)
    
    combined = hstack([synopsis_weighted, genres_sparse, tags_sparse, score_sparse, popularity_sparse])
    combined = normalize(combined)
    return combined

def build_user_profile(combined_vectors, matched_indices):
    """
    Builds a user profile vector by averaging the combined feature vectors
    of the anime that the user has watched.
    """
    user_vectors = []
    for title, idx in matched_indices.items():
        user_vectors.append(combined_vectors[idx])
    if user_vectors:
        user_profile = vstack(user_vectors)
        # Compute the mean vector along axis=0
        user_profile = user_profile.mean(axis=0)
        # Convert to a numpy array instead of a np.matrix
        user_profile = np.asarray(user_profile)
        user_profile = normalize(user_profile)
        return user_profile
    else:
        return None

def recommend_anime(user_profile, combined_vectors, df, watched_indices, top_n=10):
    """
    Recommends anime based on cosine similarity between the user profile
    and the anime vectors in the dataset. Excludes anime already watched.
    Returns the top N recommendations with titles and similarity scores.
    """
    similarities = combined_vectors.dot(user_profile.T).flatten()
    recommendations = []
    for idx, sim in enumerate(similarities):
        if idx in watched_indices:
            continue
        recommendations.append((idx, sim))
    recommendations = sorted(recommendations, key=lambda x: x[1], reverse=True)[:top_n]
    results = []
    for idx, sim in recommendations:
        title = df.loc[idx, 'title_romaji'] or df.loc[idx, 'title_english'] or df.loc[idx, 'title_native']
        results.append({'title': title, 'similarity': sim})
    return results

def main(xml_file, csv_file, top_n=10):
    # Parse user watchlist from XML (only those with watched episodes > 0)
    watchlist = parse_watchlist(xml_file)
    print(f"User watchlist (episodes watched > 0): {watchlist}")
    
    # Load the Kaggle anime dataset
    df = load_anime_dataset(csv_file)
    
    # Fuzzy matching: map watchlist titles to dataset indices
    matched_indices = match_watchlist_titles(watchlist, df)
    watched_indices = list(matched_indices.values())
    
    # Process features from the dataset
    tfidf_matrix, _ = process_description(df)
    genres_encoded, _ = process_genres(df)
    tags_encoded, all_tags = process_tags(df)
    score_scaled, _ = process_numeric_feature(df, 'averageScore')
    popularity_scaled, _ = process_numeric_feature(df, 'popularity')
    
    # Combine and normalize all features into a single vector per anime
    combined_vectors = combine_features(tfidf_matrix, genres_encoded, tags_encoded, score_scaled, popularity_scaled)
    
    # Build the user profile vector based on watched anime
    user_profile = build_user_profile(combined_vectors, matched_indices)
    if user_profile is None:
        print("No watched anime found in dataset. Cannot build user profile.")
        return
         
    # Get recommendations based on cosine similarity
    recommendations = recommend_anime(user_profile, combined_vectors, df, watched_indices, top_n=top_n)
    
    print("\nTop Recommendations:")
    for rec in recommendations:
        print(f"Title: {rec['title']}, Similarity Score: {rec['similarity']:.4f}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python script.py <watchlist_xml_file> <anime_dataset_csv_file> [top_n]")
    else:
        xml_file = sys.argv[1]
        csv_file = sys.argv[2]
        top_n = int(sys.argv[3]) if len(sys.argv) > 3 else 10
        main(xml_file, csv_file, top_n)
