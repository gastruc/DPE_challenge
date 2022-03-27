import pandas as pd
import os
from tqdm import tqdm
from urllib.request import urlretrieve

URL = 'https://koumoul.com/data-fair/api/v1/datasets/dpe-logements/data-files/DPE_logements.csv'
PATH = 'data/'
FILENAME = 'DPE_lDPE_logements.csv'

class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_data(url, output_path):
    if not os.path.exists('data/'):
        os.makedirs('data')
    if not os.path.exists(output_path):
        print('Downloading data...')
        with DownloadProgressBar(unit='B', unit_scale=True,
                                 miniters=1, desc=url.split('/')[-1]) as t:
            urlretrieve(url, filename=output_path, reporthook=t.update_to)
        print('Done')

def train_test_split(path, filename):
    df = pd.read_csv(path + filename, sep=';')
    df.drop(columns=['estimation_ges','classe_estimation_ges'],inplace=True)
    df.loc[:2000000].to_csv(path + "DPE_train.csv")
    df.classe_consommation_energie.value_counts()
    # discard misclassified elements (we cannot know if the class is right but the energy consumption is wrong and vice versa)
    wrong_class_A=(df.classe_consommation_energie=='A')&((df.consommation_energie<0) | (df.consommation_energie>50))
    df=df.loc[~wrong_class_A]

    wrong_class_B=(df.classe_consommation_energie=='B')&((df.consommation_energie<51) | (df.consommation_energie>90))
    df=df.loc[~wrong_class_B]

    wrong_class_C=(df.classe_consommation_energie=='C')&((df.consommation_energie<91) | (df.consommation_energie>150))
    df=df.loc[~wrong_class_C]

    wrong_class_D=(df.classe_consommation_energie=='D')&((df.consommation_energie<151) | (df.consommation_energie>230))
    df=df.loc[~wrong_class_D]

    wrong_class_E=(df.classe_consommation_energie=='E')&((df.consommation_energie<231) | (df.consommation_energie>330))
    df=df.loc[~wrong_class_E]

    wrong_class_F=(df.classe_consommation_energie=='F')&((df.consommation_energie<331) | (df.consommation_energie>450))
    df=df.loc[~wrong_class_F]

    wrong_class_G=(df.classe_consommation_energie=='G')&((df.consommation_energie<450))
    df=df.loc[~wrong_class_G]

    df = df.loc[df.classe_consommation_energie != 'N']
    df.drop(columns=["consommation_energie"], inplace=True)
    df.loc[2000000:].to_csv(path + "DPE_test.csv")

if __name__ == '__main__':
    download_data(URL, PATH + FILENAME)
    train_test_split(PATH, FILENAME)