import urllib.request
import os
from tqdm import tqdm

URL = 'https://koumoul.com/data-fair/api/v1/datasets/dpe-logements/data-files/DPE_logements.csv'
OUTPUT_PATH = 'data/DPE_logements.csv'

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
            urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)
        print('Done')
        
if __name__ == '__main__':
    download_data(URL, OUTPUT_PATH)