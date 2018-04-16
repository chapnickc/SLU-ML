import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

def build_df_mfccs(cats_mfccs,dogs_mfccs):
    # df_cats = pd.DataFrame(np.transpose(cats_mfccs[0]),index=np.ones(len(np.transpose(cats_mfccs[0])))*(0+1))
    df_cats = pd.DataFrame()
    for i in range(0,len(cats_mfccs)):
        df_temp = pd.DataFrame(np.transpose(cats_mfccs[i]),index=np.ones(len(np.transpose(cats_mfccs[i])))*(i+1))
        frames = [df_cats, df_temp]
        df_cats = pd.concat(frames)
    df_cats['Label']=0

    # df_dogs = pd.DataFrame(np.transpose(dogs_mfccs[0]),index=np.ones(len(np.transpose(dogs_mfccs[0])))*(0+len(cats_mfccs)+1))
    df_dogs = pd.DataFrame()
    for i in range(0,len(dogs_mfccs)):
        df_temp = pd.DataFrame(np.transpose(dogs_mfccs[i]),index=np.ones(len(np.transpose(dogs_mfccs[i])))*(i+len(cats_mfccs)+1))
        frames = [df_dogs, df_temp]
        df_dogs = pd.concat(frames)
    df_dogs['Label']=1

    frames = [df_cats, df_dogs]
    result = pd.concat(frames)
    result = result.sample(frac=1, random_state=42)
    return result

def split_data(cats_frames,dogs_frames,n_cats,n_dogs,test_size=0.3):
    y_cats = np.zeros(n_cats)
    y_dogs = np.ones(n_dogs)

    X_train_cats, X_test_cats, y_train_cats, y_test_cats = train_test_split(cats_frames, y_cats, test_size= test_size, random_state=42)
    X_train_dogs, X_test_dogs, y_train_dogs, y_test_dogs = train_test_split(dogs_frames, y_dogs, test_size= test_size, random_state=42)

    df_train = build_df_mfccs(X_train_cats,X_train_dogs)
    df_test = build_df_mfccs(X_test_cats,X_test_dogs)

    return df_train,df_test
