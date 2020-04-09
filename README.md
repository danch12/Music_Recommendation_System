# -Music_Recommendation_System
Music recommendation system for DJs looking for new songs



## Technologies Used
* Python 3.0
* Jupyter Notebook
* Sublime

## Packages Used
* Pandas
* Scikit-learn
* Tensorflow
* Numpy
* Itertools
* Scipy
* Dash
* Implicit
* Spotipy
* Librosa
* Pydub
* Matplotlib
* os
* Selenium
* TQDM
* Requests

## Introduction

I was speaking to one of my friends the other day about how DJs find songs that fit together. He told me that the way he and all his friends do it is by just listening to hours upon hours of tracks in the hopes that something good will turn up. This seemed like a very inefficient way of finding music so I set out to develop a system that would speed this process up. 
While developing this system I had a two main considerations. First, while we will be using matrix factorisation, because DJ's can sometimes want to use niche songs that basically no one has listened to we cannot totally rely on this method. Therefore there needs to be some form of content based recommendation engine. Second, as I want this system to be used by a non technical audience, I need to create an easy to use interface. In this Readme I am going to first focus on using Alternating Least Squares and matrix factorisation to recommend similar songs, then will cover how I used a neural network to create a item based system for recommending songs. Finally I will go over how I created a dashboard using Dash so that this recommender can be easily used.

## Matrix Factorisation and Alternating Least Squares

The concept of matrix factorisation in recommendation systems rose to prominence during the Netflix prize in which the winner used this technique extensively. The overall idea is that every matrix X with the shape n x p can be rewritten as a dot product of three matrices. Recommendation systems take advantage of this as by factoring the User x Item matrix into a User x Feature matrix and an Item x Feature matrix we uncover underlying features that connect the user and the item. I am going to be using the apporoach stated in [Collaborative Filtering for Implicit Feedback Datasets](http://yifanhu.net/PUB/cf.pdf) by Hu, Koren and Volinsky
