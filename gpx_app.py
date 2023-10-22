import streamlit as st
import pandas as pd
import numpy as np
import gpxpy

from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
from mplsoccer import Pitch

def readData(gpx_file):

    segment = gpx_file.tracks[0].segments[0]
    coords = pd.DataFrame([{'lat': p.latitude, 
                            'lon': p.longitude, 
                            'ele': p.elevation,
                            'time': p.time} for p in segment.points])
    coords.set_index('time', drop=True, inplace=True)

    start_time, end_time = segment.get_time_bounds()
    duration = end_time - start_time

    segment.points[0].speed = 0.0
    segment.points[-1].speed = 0.0
    gpx_file.add_missing_speeds()

    coords['speed'] = [p.speed for p in segment.points]
    coords['speed'] *= 3.6

    coords.index = (coords.index - coords.index[0]).total_seconds()

    campo_l_caxias_inf_dir = np.asarray([-27.061048, -51.193657])

    posxy = pd.DataFrame()
    posxy['x'] = coords['lat'] - campo_l_caxias_inf_dir[0]
    posxy['y'] = coords['lon'] - campo_l_caxias_inf_dir[1]

    posxy['x'] = posxy['x']*111139
    posxy['y'] = posxy['y']*111139

    coords['x'] = np.asarray(posxy['x'])
    coords['y'] = np.asarray(posxy['y'])
    
    return coords

def plotHeatmap(coords):
    
    campo_l_caxias_inf_dir = np.asarray([-27.061048, -51.193657])

    posxy = pd.DataFrame()
    posxy['x'] = coords['lat'] - campo_l_caxias_inf_dir[0]
    posxy['y'] = coords['lon'] - campo_l_caxias_inf_dir[1]

    posxy['x'] = posxy['x']*111139
    posxy['y'] = posxy['y']*111139

    pitch = Pitch(pitch_type='custom', line_zorder=2,
                pitch_color='grass', line_color='white',
                stripe_color='#c2d59d',
                pitch_length=100,pitch_width=65)
    fig, ax = pitch.draw(figsize=(6.6, 4.125))
    bin_statistic = pitch.bin_statistic(posxy['y'].values, posxy['x'].values, statistic='count', bins=(50, 50))
    bin_statistic['statistic'] = gaussian_filter(bin_statistic['statistic'], 1)
    pcm = pitch.heatmap(bin_statistic, ax=ax, cmap='hot', shading='flat')
    # fig.show()
    st.pyplot(fig)

def plotSpeeds(df_speeds, threshold):    
    
    y = np.asarray(df_speeds['speed'])
    x = np.asarray(df_speeds.index)

    mascara = y > threshold

    fig, ax = plt.subplots()
    ax.plot(x, y, linestyle='-', color='b')
    ax.plot(np.where(mascara)[0], y[mascara], 'ro')  # Marca os pontos acima de 6 com marcadores vermelhos
    ax.axhline(y=threshold, color='black', linestyle='--')

    ax.set_xlabel('Segundos')  # Corrigido de xlabel para set_xlabel
    ax.set_ylabel('Velocidade')  # Corrigido de ylabel para set_ylabel
    ax.grid(True)
    st.pyplot(fig)

def evaluateSprints(coords):
        
    distances = np.zeros(len(coords.index))
    for i in range(len(coords.index)):
        distances[i] = np.sqrt((np.asarray(coords['x'])[i] - np.asarray(coords['x'])[i-1])**2 + (np.asarray(coords['y'])[i] - np.asarray(coords['y'])[i-1])**2)
    
    indices = distances > 10
    distances[indices] = 0

    sprints = []
    starts = []
    ends = []
    for i in range(len(distances)):
        window = 10

        t_start = i
        t_end = i + window

        sprints.append(np.sum(distances[t_start:t_end]))
        starts.append(t_start)
        ends.append(t_end)

    all_sprints = pd.DataFrame()
    all_sprints['distance_sum'] = np.asarray(sprints)
    all_sprints['t_start'] = np.asarray(starts)
    all_sprints['t_end'] = np.asarray(ends)

    sprints = all_sprints.groupby(all_sprints.index // 15).apply(lambda x: x.loc[x['distance_sum'].idxmax()])
    sprints.reset_index(drop=True, inplace=True)

    sprints_sorted = sprints.sort_values(by='distance_sum', ascending=False)

    return sprints_sorted

def plotSprint(t_start):

    y = np.array(coords['x'])[t_start:t_start+10]
    x = np.array(coords['y'])[t_start:t_start+10]

    pitch = Pitch(pitch_type='custom', line_zorder=2,
                pitch_color='grass', line_color='white',
                stripe_color='#c2d59d',
                pitch_length=100,pitch_width=65)
    
    fig, ax = pitch.draw(figsize=(6.6, 4.125))
    pitch.plot(x, y, ax=ax, linewidth=4, color = 'lime')
    pitch.scatter(x[0], y[0], color='blue', label='Início', zorder=5, s=80, ax=ax)
    pitch.scatter(x[-1], y[-1], color='red', label='Fim', zorder=5, s=80, ax=ax)
    ax.legend()
    st.pyplot(fig)

st.title("GPX Data Football")

uploaded_file = st.file_uploader("Choose a GPX file", type=["gpx"])

if uploaded_file is not None:
    gpx_file = gpxpy.parse(uploaded_file)

    st.success("File uploaded successfully!")

    coords = readData(gpx_file)
    coords.index = np.arange(0,len(coords),1)

    st.write("Heatmap")
    plotHeatmap(coords)

    st.write("Speeds")
    plotSpeeds(coords,15)

    st.write("Top 5 Sprints")
    sprints = evaluateSprints(coords)
    sprints['mean_speed'] = 3.6*np.asarray(sprints['distance_sum'])/(np.asarray(sprints['t_end']) - np.asarray(sprints['t_start']))
    top_5_sprints = np.array(sprints['t_start'])[0:5]
    
    for sprint in top_5_sprints:
        plotSprint(int(sprint))

    st.write(sprints.head(5))


else:
    st.info("Please upload a GPX file.")
