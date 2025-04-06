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

# watchlist from xml
def parse_watchlist(xml_file):
  tree = ET.parse(xml_file)
  root = tree.getroot()
  watchlist = []
  for anime in root.findall('anime'):
    watched_elem = anime.find('my_watched_episodes')
    try:
      watched_count = int(watched_elem.text.strip()) if watched_elem is not None and watched_elem.text else 0
    except ValueError:
      watched_count = 0
    if watched_count > 0:
      title_elem = anime.find('series_title')
      score_elem = anime.find('my_score')
      if title_elem is not None and title_elem.text:
        try:
          user_score = float(score_elem.text.strip()) if score_elem is not None and score_elem.text else 0
        except ValueError:
          user_score = 0
        watchlist.append((title_elem.text.strip(), user_score))
  return watchlist


# load dataset from csv
def load_anime_dataset(csv_file):
    df = pd.read_csv(csv_file)
    required_columns = ['genres', 'averageScore', 'popularity', 'description', 'tags',
                        'title_romaji', 'title_english', 'title_native']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
        if col in ['averageScore', 'popularity']:
            df[col] = df[col].fillna(0)
        else:
            df[col] = df[col].fillna('')
    return df

# fuzzy matching titles to any of the three title columns
def match_watchlist_titles(watchlist, df, threshold=70):
  matched_indices = {}
  for title, user_score in watchlist:
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
      matched_indices[title] = (best_idx, user_score)
  return matched_indices


# process genres
def process_genres(df):
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

# process tags
def process_tags(df):
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
    
    all_tags = set()
    for d in tags_dicts:
        for tag in d.keys():
            all_tags.add(tag)
    all_tags = sorted(list(all_tags))
    
    tags_vectors = []
    for d in tags_dicts:
        vector = [d.get(tag, 0) for tag in all_tags]
        tags_vectors.append(vector)
    
    tags_encoded = np.array(tags_vectors)
    return tags_encoded, all_tags

# process description using TF-IDF
def fit_tfidf_vectorizer(df, column_name):
    tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    tfidf_matrix = tfidf_vectorizer.fit_transform(df[column_name].fillna(""))
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), index=df.index)
    tfidf_df = pd.DataFrame(normalize(tfidf_df), index=tfidf_df.index, columns=tfidf_df.columns)
    return tfidf_df

# process numeric features like score and popularity
def process_numeric_feature(df, column_name):
    scaler = MinMaxScaler()
    values = df[[column_name]].replace('', 0).astype(float).fillna(0)
    scaled = scaler.fit_transform(values)
    return scaled, scaler

# normalize vectors
def normalize_vector(vec):
    vec = np.array(vec, dtype=np.float32)
    n = np.linalg.norm(vec)
    return vec / n if n > 0 else vec

weights = {
    'synopsis': 0.1,
    'genres': 0.2,
    'tags': 0.3,
    'score': 0.2,
    'popularity': 0.2
}

# combine features into a single vector
def combine_features(row, weights, penalty=0.5):
    syn_vec = description_tfidf_df.loc[row.name].values
    syn = normalize_vector(syn_vec)
    if np.linalg.norm(syn_vec) == 0:
        syn = syn * penalty

    genres_vec = np.array(row['genre_vector'], dtype=np.float32)
    genres = normalize_vector(genres_vec)
    if np.linalg.norm(genres_vec) == 0:
        genres = genres * penalty

    tags_vec = np.array(row['tags_vector'], dtype=np.float32)
    tags = normalize_vector(tags_vec)
    if np.linalg.norm(tags_vec) == 0:
        tags = tags * penalty

    score = np.array([row['averageScore_scaled']], dtype=np.float32)
    popularity = np.array([row['popularity_scaled']], dtype=np.float32)

    combined = np.concatenate([
        weights['synopsis'] * syn,
        weights['genres'] * genres,
        weights['tags'] * tags,
        weights['score'] * score,
        weights['popularity'] * popularity
    ])
    return combined

# anime data processing
anime_csv_path = './ds/anime_data.csv'
anime_df = pd.read_csv(anime_csv_path)
anime_df['id'] = anime_df['id'].astype(str)

description_tfidf_df = fit_tfidf_vectorizer(anime_df, 'description')

genres_encoded, mlb = process_genres(anime_df)
anime_df['genre_vector'] = list(genres_encoded)

tags_encoded, _ = process_tags(anime_df)
anime_df['tags_vector'] = list(tags_encoded)

score_scaled, _ = process_numeric_feature(anime_df, 'averageScore')
popularity_scaled, _ = process_numeric_feature(anime_df, 'popularity')
anime_df['averageScore_scaled'] = score_scaled
anime_df['popularity_scaled'] = popularity_scaled

combined_vectors = []
for idx, row in anime_df.iterrows():
    combined_vectors.append(combine_features(row, weights))
if not combined_vectors:
    print("No combined vectors were generated. Check your feature processing steps.")
    sys.exit(1)
combined_vectors = np.vstack(combined_vectors)

# combined_vectors = [csr_matrix(combine_features(row, weights)) for idx, row in anime_df.iterrows()]
# combined_vectors = vstack(combined_vectors)


# build user profile
def build_user_profile(combined_vectors, matched_indices):
  user_vectors = []
  for title, (idx, user_score) in matched_indices.items():
    vec = combined_vectors[idx]
    weight = user_score / 10.0
    weighted_vec = vec * weight
    sparse_vec = csr_matrix(weighted_vec)
    user_vectors.append(sparse_vec)

  if user_vectors:
    user_profile = vstack(user_vectors)
    user_profile = user_profile.mean(axis=0)
    user_profile = np.asarray(user_profile)
    user_profile = normalize(user_profile)
    return user_profile
  else:
    return None


# uses cosine similarity to find similar anime, excluding watched ones
def recommend_anime(user_profile, combined_vectors, df, watched_indices, top_n=10):
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
    watchlist = parse_watchlist(xml_file)
    print(f"User watchlist (episodes watched > 0): {watchlist}")
    
    df = load_anime_dataset(csv_file)
    
    matched_indices = match_watchlist_titles(watchlist, df)
    watched_indices = list(matched_indices.values())
    
    genres_encoded, _ = process_genres(df)
    df['genre_vector'] = list(genres_encoded)
    
    tags_encoded, _ = process_tags(df)
    df['tags_vector'] = list(tags_encoded)
    
    score_scaled, _ = process_numeric_feature(df, 'averageScore')
    popularity_scaled, _ = process_numeric_feature(df, 'popularity')
    df['averageScore_scaled'] = score_scaled
    df['popularity_scaled'] = popularity_scaled
    
    global description_tfidf_df
    description_tfidf_df = fit_tfidf_vectorizer(df, 'description')
    
    combined_vectors = []
    for idx, row in df.iterrows():
        if 'genre_vector' not in row or 'tags_vector' not in row:
            continue
        combined_vectors.append(combine_features(row, weights))
    if not combined_vectors:
        print("No combined vectors were generated. Check your feature processing steps.")
        sys.exit(1)
    combined_vectors = np.vstack(combined_vectors)
    
    user_profile = build_user_profile(combined_vectors, matched_indices)
    if user_profile is None:
        print("No watched anime found in dataset. Cannot build user profile.")
        return
         
    # use consine similarity to find similar anime
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
