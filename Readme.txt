Fashion Recommender System :

based on Reverse Image search 
used in Google, Pinterst, Amazon ...

Steps in building model : 

Model used : ResNet ==> ImageNet (CNN Model) 

1) Import Model 

2) Extract Features 
    (No of trained Images , 2048 )       # (1,2048) size of vector for features extracted for one Image

3) export features 

output : n vectors of size (1,2048) in n dimension space : features extracted from all images

4) generate recommendations : 

upload image ==> generate feature vector and compare it with already generated features from trained Images. 
based on Euclidean distance, recommender closest features 


















 
      

