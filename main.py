import pickle
import PIL.Image
import cv2 as cv
import pandas as pd
import base64
from sklearn.neighbors import NearestNeighbors
import numpy as np
import streamlit as st
import PIL
import joblib

@st.cache(allow_output_mutation=True)
def loadModel():
    loaded_model = joblib.load(open('lab10.sav', 'rb'))
    return loaded_model

@st.cache(allow_output_mutation=True)
def load_bases():
    emb_db = './embedBaseDf2.csv'
    emb_base = pd.read_csv(emb_db, delimiter=',')
    emb_base['embed'] = emb_base['embed'].apply(
        lambda x: np.frombuffer(base64.b64decode(bytes(x[2:-1], encoding='ascii')), dtype=np.float32))
    return emb_base


def encodeImage(arr) -> np.ndarray:
    emds = []
    for i in arr:
        emb, _ = np.histogram(i, 2048)
        emds.append(emb)
    return emds

def build_histogram(descriptor_list, cluster_alg):
    histogram = np.zeros(2048)
    cluster_result = cluster_alg.predict(descriptor_list.astype(float))
    unique, counts = np.unique(cluster_result, return_counts=True)
    s = counts.sum()
    histogram[unique] = counts / s
    return histogram.astype('float32')

def workWithDb(img, classifier, db, db_neighbours):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    sift = cv.SIFT_create()
    _, des = sift.detectAndCompute(gray, None)
    #prediction = classifier.predict(des.astype('float'))
    predhist = build_histogram(des, classifier)
    distances, indices = db_neighbours.kneighbors(predhist.reshape(1, -1), return_distance=True)
    print(indices)
    return db.loc[indices[0], ['path']].values, distances[0]

def main():
    loaded_model = load_model()
    database = load_bases()
    db_neighbours = NearestNeighbors(n_neighbors=5, metric='cosine')
    # print(type(database))
    emdbs = encodeImage(database['embed'].values)
    db_neighbours.fit(emdbs, y=list(range(len(database))))
    up_file = st.file_uploader('Choice file', type=['jpeg', 'jpg', 'webp', 'png', 'tiff'])
    if up_file is not None:
        try:
            img = PIL.Image.open(up_file)
            img = PIL.ImageOps.exif_transpose(img)
            img_paths, dists = workWithDb(np.array(img), loaded_model, database, db_neighbours)
            print(img_paths)
            for i in range(len(img_paths)):
                st.image(PIL.Image.open(img_paths[i][0]),
                         caption='Image {} with dist {}'.format(i + 1, f'{dists[i]:.3f}', width=580))
        except Exception as e:
            st.write('CRASHED:{}'.format(e))

# main()
loaded_model = loadModel()
database = load_bases()
db_neighbours = NearestNeighbors(n_neighbors=5, metric='cosine')
#print(type(database))
emdbs = encodeImage(database['embed'].values)
db_neighbours.fit(emdbs)
if __name__ == '__main__':
    main()