"""
Prepares the data we use by figuring out all of the batted ball events, correct IDs, correct primary pos, etc. from 2018-2025
Uses Statcast pitch by pitch data and Fangraphs lookup stats data
"""

import pandas as pd
import numpy as np
from pybaseball import playerid_reverse_lookup, fielding_stats, batting_stats
import dask.dataframe as dd
import re
import unicodedata

import warnings
warnings.filterwarnings('ignore')

def load_data_prep():
    # all of the statcast regular season data from 2018-2025 (~1 min)
    data = dd.read_parquet('../data/statcast_years/*.parquet')
    statcast_data = data.query('game_type == "R"').reset_index(drop=True)
    statcast_dask_data = statcast_data.compute()

    # (~ 50 sec)
    def normalize_name(s):
        """
        normalize names: lowercase, remove spaces after periods, strip accents
        """
        s = s.lower()
        s = re.sub(r"\.\s*", ".", s)
        s = unicodedata.normalize('NFKD', s)
        return ''.join(c for c in s if not unicodedata.combining(c))

    ## we only care about the batted balls that have observed physics
    statcast_dask_data = statcast_dask_data[((~statcast_dask_data['launch_speed'].isna()) 
                                            & (~statcast_dask_data['launch_angle'].isna())
                                            & (statcast_dask_data['description'] == 'hit_into_play'))
                                            ].reset_index(drop=True).copy()

    # finds the total number of BIPs each batter has had in each season
    batter_bips = (statcast_dask_data
                .groupby(['batter', 'game_year'])['description']
                .apply(lambda s: (s == 'hit_into_play').sum())
                ).reset_index(name='events').copy()

    # finds the positions of all players
    years = np.arange(2018, 2026)
    position_dfs = []
    for yr in years:
        df = fielding_stats(yr, qual=1, split_seasons=True)
        position_dfs.append(df)
    position_df = pd.concat(position_dfs, ignore_index=True)
    position_df['Name'] = position_df['Name'].apply(normalize_name)

    # finds each player's primary position in each season (where they played the most games)
    position_df = (position_df.groupby(['IDfg', 'Name', 'Season'], as_index=False)
                .apply(lambda x: x.loc[x['G'].idxmax(), 'Pos'])
                .rename(columns={None: 'primary_pos'})
                )

    # the ids of each player + normalizing the name col
    player_ids = playerid_reverse_lookup(batter_bips['batter'].unique().tolist(), key_type='mlbam').copy()
    player_ids['name'] = player_ids['name_first'] + ' ' + player_ids['name_last']
    player_ids['name'] = player_ids['name'].apply(normalize_name)

    # normalizes the name for a few edge cases
    player_ids['name'] = player_ids['name'].apply(lambda x: 'victor mesa jr.' if x=='victor mesa' 
                                                else 'robert hassell iii' if x == 'robert hassell' 
                                                else 'dashawn keirsey jr.' if x == 'dashawn keirsey' 
                                                else x)
    player_ids.loc[player_ids['key_mlbam'] == 691777, 'name'] = 'max muncy (2)'
    player_ids.loc[player_ids['key_mlbam'] == 571970, 'name'] = 'max muncy (1)'
    print('done finding pos + normalizing name')

    # want to merge the position to the player --> need to figure out id mappings
    # all of the ids from fangraphs
    fangraph_ids = (position_df[['Name', 'IDfg']]
                    .groupby(['Name', 'IDfg'])
                    .first()
                    .reset_index()
                    )
    # deals with the max muncy problem
    fangraph_ids.loc[fangraph_ids['IDfg'] == 13301, 'Name'] = 'max muncy (1)'
    fangraph_ids.loc[fangraph_ids['IDfg'] == 29779, 'Name'] = 'max muncy (2)'

    # merged the mlbam ids with the fangraphs ids
    all_ids = (player_ids
            .merge(fangraph_ids, left_on=['key_fangraphs'], right_on=['IDfg'], how='left')
            )

    # these are the rookies for the most part that don't have fangraphs ids --> found them by their name rather than their id
    null_ids = (all_ids[all_ids['IDfg'].isna()]
                .drop(columns=['Name', 'IDfg'])
                .merge(fangraph_ids, left_on=['name'], right_on=['Name'], how='left')
                .reset_index(drop=True)
                )

    # keeping the names that weren't rookies
    all_ids = (all_ids
            .dropna(subset=['Name', 'IDfg'])
            .reset_index(drop=True)
            ).copy()

    # recombined the rookies and the non-rookies
    all_player_ids = (pd.concat([all_ids, null_ids])[['name', 'Name', 'key_mlbam', 'key_fangraphs', 'IDfg']]
                    .reset_index(drop=True)
                    ).copy()

    # dropping the remaining players because for the most part, these guys are no longer in the league (some retired, different leagues, etc.)
    all_player_ids = all_player_ids[all_player_ids['IDfg'].notna()].reset_index(drop=True).copy()
    print('done cleaning the ids')

    # this finds the primary pos for each player in each season
    cols_to_keep = ['IDfg', 'Season', 'primary_pos']
    player_primary_pos = batter_bips.merge(all_player_ids, left_on='batter', right_on='key_mlbam', how='left').copy()
    player_primary_pos = player_primary_pos[player_primary_pos['name'].notna()].copy() # gets rid of the players i got rid of in all_player_ids
    player_primary_pos = (player_primary_pos
                          .merge(position_df[cols_to_keep], left_on=['IDfg', 'game_year'], right_on=['IDfg', 'Season'], how='left')
                          .copy()
                          ) # adds primary pos

    # deals with the full-time DHs
    player_primary_pos['Season'] = player_primary_pos['Season'].fillna(player_primary_pos['game_year'])
    player_primary_pos['primary_pos'] = player_primary_pos['primary_pos'].fillna('DH')
    player_primary_pos['IDfg'] = player_primary_pos['IDfg'].astype(int)

    # deals with ohtani
    player_primary_pos.loc[(player_primary_pos['name'] == 'shohei ohtani'), 'primary_pos'] = 'DH'

    # filters it to only position players (no pitchers) & deals with matt davidson
    matt_davidson = player_primary_pos['name'] == 'matt davidson'
    year_2020 = player_primary_pos['game_year'] == 2020
    player_primary_pos.loc[(matt_davidson) & (year_2020), 'primary_pos'] = '1B'
    player_primary_pos = (player_primary_pos[player_primary_pos['primary_pos'] != 'P']
                          .reset_index(drop=True)
                          .copy()
                          )
    player_primary_pos = (player_primary_pos
                          .sort_values(by=['name', 'game_year'], ascending=[True, True])
                          .reset_index(drop=True)
                          .copy()
                          )

    # adding age to each season for each player (~1 min)
    player_ages = batting_stats(2018, 2025, qual=1)[['IDfg', 'Season', 'Age']].copy()
    player_primary_pos = (player_primary_pos
                          .merge(player_ages, left_on=['IDfg', 'game_year'], right_on=['IDfg', 'Season'], how='left')
                          .copy())
    player_primary_pos = player_primary_pos[['game_year', 'batter', 'IDfg', 'name', 'Age', 'primary_pos', 'events']].copy()

    # adding PA to each season for each player (~ 2 min)
    pa_cols = ['IDfg', 'Season', 'PA']
    player_appearences = batting_stats(2018, 2025, qual=0)
    player_appearences = player_appearences[['IDfg', 'Season', 'Name', 'Team', 'PA']]
    player_appearences.loc[player_appearences['IDfg'] == 13301, 'Name'] = 'max muncy (1)'
    player_appearences.loc[player_appearences['IDfg'] == 29779, 'Name'] = 'max muncy (2)'
    player_primary_pos = (player_primary_pos
                          .merge(player_appearences[pa_cols], left_on=['IDfg', 'game_year'], right_on=['IDfg', 'Season'], how='left')
                          .drop(columns=['Season'])
                          )
    print('done adding age + dealing with edge cases')

    batted_ball_cols = ['game_year', 'game_date', 'batter', 'events', 'stand', 'home_team', 'away_team', 'bb_type', 'launch_speed', 'launch_angle']
    batted_ball_events = statcast_dask_data[batted_ball_cols]
    batted_ball_events['is_HR'] = (batted_ball_events['events'] == 'home_run').astype(int)

    ## filters it to only look at the batted balls that have an id in player_primary_pos --> excludes pitcher batted balls, etc.
    batted_ball_events = (batted_ball_events[batted_ball_events['batter'].isin(list(player_primary_pos['batter'].unique()))]
                          .reset_index(drop=True)
                          .copy()
                          )

    # downloading as parquets so that I can read them in faster
    batted_ball_events.to_parquet('../data/batted_ball_events.parquet', index=False)
    player_primary_pos.to_parquet('../data/player_primary_pos.parquet', index=False)

if __name__ == "__main__":
    load_data_prep()
    print('run complete!')