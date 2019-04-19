import pandas as pd
NUM_POP_LYRICS = 30000
NUM_ROCK_LYRICS = 50000
NUM_HIP_HOP_LYRICS = 20000
RANDOM_SEED = 50

CO_VARIATES = ['Pop', 'Rock', 'Hip Hop']
artists = pd.read_csv('../../../data/artists-data.csv')
transformed_lyrics = pd.read_csv('../../../data/lyrics-data.csv')

joined_dataset = transformed_lyrics.merge(artists, left_on='ALink', right_on='Link')
filtered_dataset = joined_dataset[
    (joined_dataset['Genre'].isin(CO_VARIATES)) & (joined_dataset['Idiom'] == 'ENGLISH')]

pop_df = filtered_dataset[filtered_dataset['Genre'] == CO_VARIATES[0]].sample(NUM_POP_LYRICS, random_state=RANDOM_SEED)
rock_df = filtered_dataset[filtered_dataset['Genre'] == CO_VARIATES[1]].sample(NUM_ROCK_LYRICS,
                                                                               random_state=RANDOM_SEED)
hip_hop_df = filtered_dataset[filtered_dataset['Genre'] == CO_VARIATES[2]].sample(NUM_HIP_HOP_LYRICS,
                                                                                  random_state=RANDOM_SEED)

frames = [pop_df, rock_df, hip_hop_df]

dataset = pd.concat(frames)

dataset = dataset[['Lyric', 'Genre', 'Artist']]

dataset.to_csv('../../../data/dataset-big.csv', index=None, header=True)
