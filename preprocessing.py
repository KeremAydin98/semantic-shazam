import pandas as pd
import os

if "combined-data.pkl" not in os.listdir("Data/"):

    # Load the csv data
    df_artist = pd.read_csv("Data/artists-data.csv")
    df_lyrics = pd.read_csv("Data/lyrics-data.csv")

    # Rename the linking column to connect both dataframes
    df_artist = df_artist.rename(columns={"Link":"ALink"})

    # Merge the lyrics and artists dataframes
    df = df_artist.merge(df_lyrics, on="ALink")

    # Filter to only get English songs
    df = df[df["language"] == "en"]

    # Drop the unnessecary columns
    df = df.drop(["Artist","Songs","Popularity", "ALink", "SLink"], axis=1)

    # Keep only the first genres
    df["Genres"] = df["Genres"].map(lambda x: str(x).split(";")[0])

    # Remove the nan genre rows
    df = df[df["Genres"] != "nan"]

    # Drop the none values
    df = df.dropna()

    # Pickle to use the dataframe later
    df.to_pickle("Data/combined-data.pkl")

else:

    # Reading the dataframe if it is already created
    df = pd.read_pickle("Data/combined-data.pkl")




