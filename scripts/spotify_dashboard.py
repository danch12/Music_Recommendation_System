import dash
import dash_core_components as dcc
from dash.dependencies import Input, Output, State
import dash_html_components as html
import dash_table

import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
import plotly.graph_objs as go
import tensorflow as tf
import pickle
import config
from tensorflow import keras
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import initializers
from tensorflow.keras import optimizers
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Conv2D,AveragePooling2D, BatchNormalization,Dropout,Flatten,Dense
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Model
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import spotipy.util as util
import seaborn as sns
from sklearn.metrics import pairwise_distances_argmin_min, pairwise_distances


import io
import librosa
import librosa.display
import soundfile as sf
import glob

from implicit.als import AlternatingLeastSquares
from scipy.sparse import coo_matrix
import pydub
from urllib.request import urlopen
import requests
from shutil import copyfileobj
from tempfile import NamedTemporaryFile
from urllib.request import urlopen, Request

df_path= '/Users/danielchow/python stuff/Spotify_similar_songs/music_values.csv'

df= pd.read_csv(df_path)
collab_df=pd.read_csv('/Users/danielchow/python stuff/song_collab_filtering/mixesdb_df_for_recs.csv')
collab_df.drop(columns=['Unnamed: 0'],inplace=True)
model_path ='/Users/danielchow/python stuff/Spotify_similar_songs/more_data_08-2.07.h5'
#load pretrained model
model = load_model(model_path)
# take the last layer off the model so we can get to the latent features
new_model =model
new_model.layers.pop()
new_model_2 = Model(new_model.input, new_model.layers[-3].output)
client_id= config.client_id
client_secret =config.client_secret

username = '1143043561'
auth = SpotifyClientCredentials(
client_id=client_id,
client_secret=client_secret
)


try:
    token = auth.get_access_token()
except:
    os.remove(f'.cache-{username}')
    token = auth.get_access_token()

#create spotify object
spotify = spotipy.Spotify(auth=token)

def spectrogram_then_latent(url, song_id, name, model):
    '''basically the same as the function i used to
    create a load of spectrograms but just for 1 this time
    takes  url and turns the preview mp3 
        into a spectrogram and then uses a model to 
        extract latent features'''   

    if url != None:
        try:
            mp3_url = url
            wav = io.BytesIO()
            with urlopen(mp3_url) as r:
                r.seek = lambda *args: None  # allow pydub to call seek(0)
                pydub.AudioSegment.from_file(r).export(wav, "wav")

            wav.seek(0)
            y, sr = librosa.load(wav)

            # mel-scaled power (energy-squared) spectrogram
            mel_spec = librosa.feature.melspectrogram(y,
                                                      sr=sr,
                                                      n_mels=128,
                                                      hop_length=1024,
                                                      n_fft=2048)
            # Convert to log scale (dB). We'll use the peak power as reference.
            log_mel_spec = librosa.amplitude_to_db(mel_spec, ref=np.max)
            #make dimensions of the array smaller
            log_mel_spec = np.resize(log_mel_spec, (128, 644))

            log_mel_spec_arr = log_mel_spec.reshape(
                log_mel_spec.shape[0], log_mel_spec.shape[1], 1)
            pre_process = np.expand_dims(log_mel_spec_arr, axis=0)
            pre_process = pre_process / 255
            latent = model.predict(pre_process)
            

        except:
            print('song has no useable url')
            return
    
        #just going to keep it as a df for ease of use even though only 1 row
        latent_df= pd.DataFrame(latent)
        latent_df=latent_df.loc[~(latent_df==0).all(axis=1)]
        latent_df['song_names']= name
        latent_df['id']= song_id
    
    else:
        print('song has no useable url')
        return
    
    return (latent_df)

def get_spotify_quals(music_df):
    '''get spotify music qualities and put into
        dataframe then merge into main dataframe'''
    #get ids of songs so can search spotify
    id_list=list(music_df['id'].values)
    #earlier function that helps look for music quals
    quals = get_music_quals(id_list)
    #put the qualities as column headers
    df=pd.DataFrame(columns=list(quals[list(quals.keys())[0]][0].keys()))
    for ind, key in enumerate(quals.keys()):
    #iterate over songs and get the spotify qualities for each song into the df
        song=quals[key]
        try:
            df.loc[ind]=list(song[0].values())
        except:
            print(song)
    #merge the two dataframes together on id and return
    orig_and_spotify =pd.merge(music_df,df,on='id',how='outer')
    
    #drop columns we dont really need
    orig_and_spotify.drop(columns=['track_href','analysis_url','duration_ms','type','uri'],inplace=True)
    #turning some of the columns into ints so can be used
    #using try and except as there may be some NANs
    try:
        orig_and_spotify.key=orig_and_spotify.key.astype(int)
        orig_and_spotify['mode']=orig_and_spotify['mode'].astype(int)
        orig_and_spotify.time_signature=orig_and_spotify.time_signature.astype(int)
        return orig_and_spotify
    except:
        return orig_and_spotify



def closest_k_nodes(node, nodes,k=5):
    node=node.reshape(1,-1)
    dist_2 = pairwise_distances(nodes,node)
    smallest_inds=np.argsort(dist_2[:,0])[:k]
    smallest_dists=dist_2[smallest_inds]
    return (smallest_inds, smallest_dists)

def find_similar(song_id,song_list=df,num_songs=5,diff_genre='no',same_key='yes',similar_tempo='yes',margin=3,scaled=False):


    song_index=song_list[song_list.id==song_id].index[0]


    orig_key = song_list.loc[song_index,'key']
    orig_tempo= song_list.loc[song_index, 'tempo']
    orig_genre= song_list.loc[song_index,'genre_approx']
    node = song_list.loc[song_index]

    #check if you want songs of same key
    if same_key=='yes':
    #if yes then filter out other keys    
        print(f'key:{orig_key}')
        song_list = song_list[song_list.key ==orig_key]
    #can also enter number to specify what key you want    
    elif type(same_key) !=str:
        song_list = song_list[song_list.key==same_key]


    # check if you want similar tempo    
    if similar_tempo=='yes':
        print(f'tempo:{orig_tempo}')
        #if yes can also specify how similar you want it 
        song_list=song_list[song_list.tempo.between(int(orig_tempo)-margin,int(orig_tempo)+margin)]

    elif type(similar_tempo) !=str:
        #can also specify a specific tempo that you want
        song_list = song_list[song_list.tempo.between(int(similar_tempo)-margin,int(similar_tempo)+margin)]

    #check if you want a different genre
    if diff_genre=='yes':
        print(f'genre:{orig_genre}')
        song_list= song_list[song_list.genre_approx != orig_genre]


    #adds requested song back into dataframe in case it was filtered out
    song_list=song_list.append(node,ignore_index=True)
    song_id_info=song_list.copy()
    #makes sure certain columns are ints as they come as strings from spotify
    song_list.key=song_list.key.astype(int)
    song_list['mode']=song_list['mode'].astype(int)
    song_list.time_signature=song_list.time_signature.astype(int)
    #exclude any other column which is a string
    song_list=song_list.select_dtypes(exclude=object)

    #scale the data
    if scaled==True:
        scaler = StandardScaler()
        song_list = scaler.fit_transform(song_list)

        node = song_list[-1]
        nodes = song_list

    else:
        node=np.asarray(song_list.iloc[-1])
        nodes=song_list

    #find closest song using euclidean distance
    closest_inds,distances = closest_k_nodes(node,nodes,num_songs)

    #get song ids of closest songs
    song_ids=song_id_info.loc[closest_inds,'id']
    #make sure there is no NANs 
    song_ids=[x for x in song_ids if str(x) != 'nan']

    #find track names using spotify API
    tracks=spotify.tracks(list(song_ids))
    song_names=[]
    song_previews=[]
    for track in tracks['tracks']:
        song_names.append((track['name'],':',track['artists'][0]['name']))
        song_previews.append(track['preview_url'])
    #create dataframe of songs and their euclidean distance from the original song
    close_song_df=pd.DataFrame({'song_ids':song_ids,
                                'song_names':song_names,
                                'song_previews':song_previews,
                                'distances':distances.flatten()})
    return (close_song_df)

def collab_filter(song_id, user_song_df, num_songs=5):
    '''
    song_id = spotify id for individual song
    user_song_df= dataframe with users, songs, playcounts etc
    for the time being i am not going to enable filtering by key/tempo as not enough songs
    but in future will do
    '''

    song_num = user_song_df[user_song_df.spotify_id == song_id].song_nums.values[0]
    print(song_num)
    print(type(song_num))
    #orig_key = song_list[song_list.spotify_id==song_id].key.values[0]
    #orig_tempo= song_list[song_list.spotify_id==song_id].tempo.values[0]

    #check if you want songs of same key
    #if same_key=='yes':
    #if yes then filter out other keys
    #    print(f'key:{orig_key}')
    #    song_list = song_list[song_list.key ==orig_key]

    #can also enter number to specify what key you want
    # elif type(same_key) !=str:
    #     song_list = song_list[song_list.key==same_key]

    # check if you want similar tempo
    #  if similar_tempo=='yes':
    #     print(f'tempo:{orig_tempo}')
    #if yes can also specify how similar you want it
    #     lower= int(orig_tempo)-margin
    #    higher=int(orig_tempo)+margin
    #    song_list=song_list[song_list.tempo.between(lower,higher)]

    #elif type(similar_tempo) !=str:
    #can also specify a specific tempo that you want
    #   song_list = song_list[song_list.tempo.between(int(similar_tempo)-margin,int(similar_tempo)+margin)]

    # refined_ids=song_list.spotify_id
    #this will be updated
    user_song_refined = user_song_df
    #[user_song_df.spotify_id.isin(
    #    refined_ids)].copy()

    plays = user_song_refined['size']
    user_nums = user_song_refined.user_nums
    song_nums = user_song_refined.song_nums

    B = coo_matrix((plays, (song_nums, user_nums))).tocsr()

    model = AlternatingLeastSquares(factors=30)
    model.fit(B)
    songs_inds = model.similar_items(song_num, N=num_songs)
    songs_inds = [tup[0] for tup in songs_inds]

    return user_song_df[user_song_df.song_nums.isin(songs_inds)]



def get_similar_for_new(song_id,
                        collab_df,
                        music_df,
                        model,
                        num_songs=5,
                        diff_genre='no',
                        same_key='yes',
                        similar_tempo='yes',
                        margin=3,
                        scaled=False):
    '''
        First checks if song in collaborative filtering database
        if not it then checks if song id in music dataframe and 
        if song is in neither tries to create the spectrogram 
        and find spotify features for the specified song, 
        then finds similar songs to that song using the 
        aforementioned qualities
    '''
    #checks if song in music dataframe

    if song_id in collab_df.spotify_id.unique():
        return collab_filter(song_id=song_id,user_song_df=collab_df,num_songs=num_songs)
    
    #connect to spotify api    
    client_id= config.client_id
    client_secret =config.client_secret

    username = '1143043561'
    auth = SpotifyClientCredentials(
    client_id=client_id,
    client_secret=client_secret
    )


    try:
        token = auth.get_access_token()
    except:
        os.remove(f'.cache-{username}')
        token = auth.get_access_token()

    #create spotify object
    spotify = spotipy.Spotify(auth=token)
    if song_id in music_df.id.unique():
        #if it is in the dataframe can just find similar songs
        close_song_df = find_similar(song_id,
                                             music_df,
                                             num_songs=num_songs,
                                             diff_genre=diff_genre,
                                             same_key=same_key,
                                             similar_tempo=similar_tempo,
                                             margin=margin,
                                             scaled=scaled)
        
        return close_song_df

    else:
        track = spotify.track(song_id)
        song_name = track['name']
        preview_url = track['preview_url']
        #check if the song has a url
        if preview_url:
            single_song_df = spectrogram_then_latent(preview_url, song_id,
                                                     song_name, model)

            #can just use the same spotify function as before
            single_song_with_spotify = get_spotify_quals(single_song_df)

            #sometimes the names of the columns can become strings/ints so need to check their the same
            cols = [str(col) for col in list(single_song_with_spotify.columns)]
            
            
            single_song_with_spotify.columns = cols
            #need to remove genre approx in the future
            single_song_with_spotify['genre_approx'] = 10
            single_song_with_spotify = single_song_with_spotify.reindex(sorted(single_song_with_spotify.columns), axis=1)
            music_df = pd.concat([music_df, single_song_with_spotify], ignore_index=True)
            #now we have song in dataframe can just do same process as the first bit of if statement
            close_song_df = find_similar(song_id,
                                                 music_df,
                                                 num_songs=num_songs,
                                                 diff_genre=diff_genre,
                                                 same_key=same_key,
                                                 similar_tempo=similar_tempo,
                                                 margin=margin,
                                                 scaled=scaled)
            return close_song_df
        
        else:
            return 'song has no url/ no spotify info'
           










external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app =dash.Dash(__name__, external_stylesheets=external_stylesheets)
#data for a graph is going to come in the form of a single dictionary or a list of dictionaries
#single dictionary would be single line/one set of bars



col_names=['song_ids','song_names','distances']
#html of entire project
app.layout  =html.Div(children=[
    html.Div(children='Diff genre, same key and tempo only work for songs not in collaborative database'),
    dcc.Input(id='input',value='song_id',type='text'),
    dcc.Input(id='input_2',value='num of songs',type='text'),
    dcc.Input(id='input_3',value='diff_genre',type='text'),
    dcc.Input(id='input_4',value='same_key',type='text'),
    dcc.Input(id='input_5',value='similar_tempo',type='text'),
    html.Div(children=[html.Table(id='table'), html.Div(id='table-output')])

    ])


@app.callback(
    Output(component_id='table-output',component_property='children'),
    [Input(component_id='input',component_property='value'),
    Input(component_id='input_2',component_property='value'),
    Input(component_id='input_3',component_property='value'),
    Input(component_id='input_4',component_property='value'),
    Input(component_id='input_5',component_property='value')
    ]
    )

def update_table(input_data1,input_data2,input_data3,input_data4,input_data5):
    try:
        num_songs = int(input_data2)
    except:
        return 'enter a integer for number of songs'

    similar_songs_df=get_similar_for_new(song_id=input_data1,
                        music_df=df,
                        collab_df=collab_df,
                        model=new_model_2,
                        num_songs=num_songs,
                        diff_genre=input_data3,
                        same_key=input_data4,
                        similar_tempo=input_data5 )
    if 'set_list' in similar_songs_df.columns.tolist():
        similar_songs_df.drop(columns=['artist','song','user_nums','song_nums','songs','size','set_title'],inplace=True)
        similar_songs_df.columns=['song_and_artist','spotify_song_name','spotify_id','spotify_preview','set_dj']
        similar_songs_df.drop_duplicates(subset=['spotify_id'],inplace=True)
    similar_songs_df.fillna('-')


    data_table = dash_table.DataTable(id='datatable-data',
                                        data = similar_songs_df.to_dict('records'),
                                        columns =[{'id': c , 'name':c} for c in similar_songs_df.columns],
                                         #fixed_rows={'headers': True, 'data': 10},
                                       # style_cell={'width': '100px'},
                                        style_table={'overflowX': 'scroll'},
                                        style_data_conditional=[ {
                                            'if': {'row_index': 'odd'},
                                             'backgroundColor': 'rgb(248, 248, 248)'}],
                                        style_header={
                                            'backgroundColor': 'rgb(230, 230, 230)',
                                            'fontWeight': 'bold'
                                        })
    #data = [a for sublist in data ]
    return data_table

   
        



if __name__ =='__main__':
    app.run_server(debug=True)