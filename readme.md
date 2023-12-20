

Copy all content of `kcal.py` to colab notebook and add one cell above with following:
```
from google.colab import drive
drive.mount('/content/drive')

!cp -r "/content/drive/MyDrive/Data_and_Labels" "/content/" # replace first path with path on your google drive to the dataset
!pip install timm wandb lightning
!wandb login
```