import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from skimage.feature import hog, local_binary_pattern
from sklearn import svm 
from sklearn import metrics

class HogClassifier:
  def __init__(self,hog_orientations=8, hog_pixels_per_cell=4, hog_cells_per_block = 1, n_components = .95, kernel = 'linear', C = 1.):
    self.hog_mean = None
    self.hog_std = None
    self.hog_pca = None
    
    self.hog_orientations=hog_orientations
    self.hog_pixels_per_cell=hog_pixels_per_cell
    self.hog_cells_per_block=hog_cells_per_block
    self.n_components = n_components
    self.kernel = kernel
    self.C = C
  
  #расчет hog признаков 
  def hog_feat(self,image):
    return hog(image, orientations=self.hog_orientations, pixels_per_cell=(self.hog_pixels_per_cell, self.hog_pixels_per_cell), cells_per_block=(self.hog_cells_per_block, self.hog_cells_per_block*3))
  
  def standardize(self,X,mean,std):
    X -= mean
    X /= std
    return X
  
  #ф-я обучения классификатора
  def fit(self,train_X,train_Y):
    def hog():
      hog_train_X = np.array(list(map(lambda x : self.hog_feat(x), train_X)))
      self.hog_mean = hog_train_X.mean(0)
      self.hog_std = hog_train_X.std(0)
      return self.standardize(hog_train_X,self.hog_mean,self.hog_std)

    self.hog_train_X = hog()

    self.hog_train_X = np.nan_to_num(self.hog_train_X)

    #метод главных компонент для HOG
    self.hog_pca = PCA(n_components=self.n_components)
    self.hog_pca.fit(self.hog_train_X)
    
    train_X = self.preprocess(train_X)
    
    #метод опорных векторов
    # self.clf = svm.SVC(C = self.C, kernel = self.kernel)
    self.clf = svm.LinearSVC(C = self.C) #, penalty='l1',dual=False)
    self.clf.fit(train_X, train_Y)      
  
  def evaluate(self,val_X, val_Y):
    predict_Y = self.predict(val_X)
    return metrics.accuracy_score(val_Y, predict_Y)
  
  def preprocess(self,test_X):
    #hog признаки
    hog_test_X = np.array(list(map(lambda x : self.hog_feat(x), test_X)))    
    #стандартизация
    hog_test_X = self.standardize(hog_test_X,self.hog_mean,self.hog_std)
    
    hog_test_X = np.nan_to_num(hog_test_X)

    #pca transform для hog
    hog_pca_test_X = self.hog_pca.transform(hog_test_X)
    
    return hog_pca_test_X
    
  def predict(self,test_X):
    test_X = self.preprocess(test_X)
    return self.clf.predict(test_X)


def load_model(model_file):
    with open(model_file, 'rb') as f:
        model = pickle.load(f)
    return model


# class HogClassifier:
#   def __init__(self,hog_orientations=8, hog_pixels_per_cell=4, n_components = .95, kernel = 'linear', C = 1.):
#     self.hog_mean = None
#     self.hog_std = None
#     self.hog_pca = None
    
#     self.hog_orientations=hog_orientations
#     self.hog_pixels_per_cell=hog_pixels_per_cell
#     self.n_components = n_components
#     self.kernel = kernel
#     self.C = C
  
#   #расчет hog признаков 
#   def hog_feat(self,image):
#     return hog(image, orientations=self.hog_orientations, pixels_per_cell=(self.hog_pixels_per_cell, self.hog_pixels_per_cell), cells_per_block=(1, 1))
  
#   def standardize(self,X,mean,std):
#     X -= mean
#     X /= std
#     return X
  
#   #ф-я обучения классификатора
#   def fit(self,train_X,train_Y):
#     def hog():
#       hog_train_X = np.array(list(map(lambda x : self.hog_feat(x), train_X)))
#       self.hog_mean = hog_train_X.mean(0)
#       self.hog_std = hog_train_X.std(0)
#       return self.standardize(hog_train_X,self.hog_mean,self.hog_std)

#     self.hog_train_X = hog()

#     self.hog_train_X = np.nan_to_num(self.hog_train_X)

#     #метод главных компонент для HOG
#     self.hog_pca = PCA(n_components=self.n_components)
#     self.hog_pca.fit(self.hog_train_X)
    
#     train_X = self.preprocess(train_X)
    
#     #метод опорных векторов
#     # self.clf = svm.SVC(C = self.C, kernel = self.kernel)
#     self.clf = svm.LinearSVC(C = self.C) #, penalty='l1',dual=False)
#     self.clf.fit(train_X, train_Y)      
  
#   def evaluate(self,val_X, val_Y):
#     predict_Y = self.predict(val_X)
#     return metrics.accuracy_score(val_Y, predict_Y)
  
#   def preprocess(self,test_X):
#     #hog признаки
#     hog_test_X = np.array(list(map(lambda x : self.hog_feat(x), test_X)))    
#     #стандартизация
#     hog_test_X = self.standardize(hog_test_X,self.hog_mean,self.hog_std)
    
#     hog_test_X = np.nan_to_num(hog_test_X)

#     #pca transform для hog
#     hog_pca_test_X = self.hog_pca.transform(hog_test_X)
    
#     return hog_pca_test_X
    
#   def predict(self,test_X):
#     test_X = self.preprocess(test_X)
#     return self.clf.predict(test_X)


# def load_model(model_file):
#     with open(model_file, 'rb') as f:
#         model = pickle.load(f)
#     return model
    
# type_clf = pickle.load(open('/content/drive/MyDrive/type_model.pkl', 'rb'))
# orient_clf = pickle.load(open('/content/drive/MyDrive/orient_model.pkl', 'rb'))
