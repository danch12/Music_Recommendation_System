# Music_Recommendation_System
Music recommendation system for DJs looking for new songs

## Contents

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

I was speaking to one of my friends the other day about how DJs find songs that fit together. He told me that the way he and all his friends do it is by just listening to hours upon hours of tracks in the hopes that something good will turn up. This seemed like a very inefficient way of finding music so I set out to develop a system that would speed this process up for him. 
While developing this system I had a two main considerations. First, while we will be using matrix factorisation, because DJ's can sometimes want to use niche songs that basically no one has listened to we cannot totally rely on this method. Therefore there needs to be some form of content based recommendation avenue. Second, as I want this system to be used by a non technical audience, I need to create an easy to use interface. In this Readme I am going to first focus on using Alternating Least Squares and matrix factorisation to recommend similar songs, then will cover how I used a neural network to create a item based system for recommending songs. Finally I will go over how I created a dashboard using Dash so that this recommender can be easily used.

## Matrix Factorisation and Alternating Least Squares

The concept of matrix factorisation in recommendation systems rose to prominence during the Netflix prize in which the winner used this technique extensively. The overall idea is that every matrix X with the shape n x p can be rewritten as a dot product of three matrices. Recommendation systems take advantage of this as by factoring the User x Item matrix into a User x Feature matrix and an Item x Feature matrix we uncover underlying features that connect the user and the item. This may make more sense in an example, take for instance Netflix recommending movies, underlying features may be things like if it is a comedy or if it has Will Smith in it (in actuality however these features will have no labels). Likewise, the users may have a preference for comedy or a preference for Will Smith. Therefore, if we can break down the items into features and breakdown our users into how much they like those features, we can recommend items with similar features to those that the user likes. For example, if we know that a user likes Will Smith and comedies from his past ratings we can recommend the Fresh Prince Of Bel Air. Using SVD to solve for the latent features can be very computationally expensive so to get around this we can use Alternating Least Squares to approximate them much quicker.I am going to be using the apporoach stated in [Collaborative Filtering for Implicit Feedback Datasets](http://yifanhu.net/PUB/cf.pdf) by Hu, Koren and Volinsky. Although I am going to use a module called implicit for implementing this on our dataset, for clarity I am going to manually code an implementation of ALS to show how the model works. What my code lacks, amoung other things, is the ability to run both the user loop and the item loop in parallel which is one of the main attractions of using ALS.

```
class rec_sys_als():
    
    def __init__(self, plays, n_factors=40, item_reg=0.1,user_reg=0.1,seed=1,n_iters=100):
        self.plays =plays
        self.num_users = plays.shape[0]
        self.num_songs = plays.shape[1]
        self.n_factors =n_factors
        self.item_reg = item_reg
        self.user_reg = user_reg
        self.n_iters=n_iters
    
    def train(self):
        #init user and item factors
        self.x = np.random.normal(0, 0.1, (self.num_users, self.n_factors))
        self.y = np.random.normal(0, 0.1, (self.num_songs, self.n_factors))
        
        #user and item reguarization 
        lambda_I_u =np.eye(self.n_factors)* self.user_reg
        lambda_I_i =np.eye(self.n_factors)* self.item_reg
        
        for num in range(self.n_iters):
            #"Before looping through all users/items, we compute the f ×f matrix YTY and XTX"
            Y_trans_Y = self.y.T.dot(self.y)
            X_trans_X =self.x.T.dot(self.x)
            Y_I = np.eye(Y_trans_Y.shape[0]) 
            X_I = np.eye(X_trans_X.shape[0]) 



            for u in range(self.num_users):
                
                
                # trying to solve this - x =(YTCuY +λI)−1YTCup(u) for each user row
                user_row=self.plays[u,:]
                CuI =np.diag(user_row)
                
                YTCuY= self.y.T.dot(CuI).dot(self.y)

                YTCupu=self.y.T.dot(CuI + Y_I).dot(user_row.T)

                self.x[u] = np.linalg.solve(Y_trans_Y + YTCuY + lambda_I_u, YTCupu)

            for i in range(self.num_songs):
                
                #trying to solve this - yi = (XT CiX + λI)−1XT Cip(i) for each item row
                song_row=self.plays[:,i]
                CiI =np.diag(song_row)

                YTCiY= self.y.T.dot(CiI).dot(self.y)

                YTCipi=self.x.T.dot(CiI + X_I).dot(user_row.T)

                self.y[i] = np.linalg.solve(X_trans_X  + YTCiY + lambda_I_i, YTCipi)
        
        return self.x,self.y
```

## Obtaining and Cleaning the Data

To be able to put this theory in practice I needed a dataset so I scraped data from [MixesDB](https://www.mixesdb.com/w/Main_Page) a website that provides track lists for thousands of DJ sets and millions of songs. This seemed like a good choice for a dataset as it avoids one of the key problems that most recommendation systems have, namely that we do not usually know why someone has listened to a song or bought a product. In this case we know almost exactly why the users in this dataset have chosen each song - they choose the song because they like it and think it sounds good.The downside is that this dataset will be very sparse so some recommendations may be dubious due to lack of user, item interaction. While cleaning the data, the most important factor was to organise the data in a may that allowed for ease of use once the project was finished so anyone could use the system. This is why I decided to add the Spoitfy song_id number and other spotify identifiers as with these added my hope was that a non technical user of this system could enter in a spotify song id, press a button and then my system could populate a spotify playlist which can then be easily accessed. One secondary benefit is that it allows access to the song analysis and also a preview of the track which will be utilized later in the project.

## Modelling the Data

As Alluded to above,  we will be using the Alternating Least Squares model which has been implemented in the [Implicit](https://implicit.readthedocs.io/en/latest/) module. The issue with finding similar songs is that there isn't really a simple metric that can rate how similar two songs are (although we try to develop a system for doing this in the next part of the project). Therefore it can be hard to rate how well this model performs or how to best tune its hyper parameters. Additionally as most artists only appeared a couple of times we could not use the models ability to find songs played by the same artist. Therefore the only solution was to just listen to the songs the model suggested with the people who were going to be using the system and see if they agreed with the choices the model made. In the future I would like to develop a better method for assessing the model performance.

## Unknown songs

Although the above method works well for when we look for songs that are found in our mixesdb dataset, if a song is not found in it then the model cannot give recommendations. Therefore we need another solution for finding similar songs. The method that I am going to use is inspired by this [article](https://benanne.github.io/2014/08/05/spotify-cnns.html) which detailed one way that Spotify deals with their cold start problem - that new or unpopular songs will not have enough user data to be able to give recommendations. The basic idea is that you can turn an MP3 into a spectrogram , then use a convolutional neural net to classify the spectrogram into a specific genre. Once you have trained the neural net to do this you can then take the last layer off the neural net and extract latent features of the songs. After we obtain the latent features as numerical values we can use the Euclidean distance between each song in the dataset to determine what the most similar songs are.
Before jumping into the actual process I will explain what Spectrograms are and give a quick overview on how convoluted neural networks work.

## Spectrograms
![spectrogram of a rock song](https://i.imgur.com/ouW2vjz.png)

Spectrograms are basically visual representations of sound. They represent signal loudness over time at various frequencies. Time runs from left to right and the vertical axis represents frequency, which can also be thought of as pitch or tone, with the lowest frequencies at the bottom and the highest frequencies at the top.  The amplitude (or energy or “loudness”) of a particular frequency at a particular time is represented by the third dimension, color, with dark blues corresponding to low amplitudes and brighter colors up through red corresponding to progressively stronger (or louder) amplitudes.

### Convoluted Neural Networks

For the purposes of this readme I will assume that the reader knows how a basic neural network works and focus primarily on how a convoluted neural network differs. 

The most important feature in a convoluted neural network is the convolutional layer- meaning that neutrons in the first convolutional layer are not connected to every pixel in an image but rather to those in their receptive field (a receptive field is basically a set area in which the neutron looks at). Following this, each neutron in the second convolutional layer is only connected to neutrons in a smaller rectangle within the first layer and so on. This structure allows the CNN to focus on low level parts (such as lines and curves) of the image in the first few layers and then builds these low level parts into higher level features. A neuron located in row r, column c of a given layer is connected to the outputs of the neurons in the previous layer located in rows r to r + fh – 1, columns c to c + fw – 1, where fh and fw are the height and width of the receptive field. A commonly used technique used to reduce computational strain is spacing out the receptive fields by moving slightly further along the image before collecting the image data again. This is called the stride, so for example if our CNN had a stride of 2, the receptive fields would be spaced out to be two pixel columns apart.

So how do these neurons ‘look’ at the image? Each neuron’s weights (called filters) can be represented as a small image the size of the receptive field. So a filter that looks for vertical lines would be a black square with a vertical white line going down the middle ( a square of 0s with a line of 1s down the middle) or a horizontal filter would be a black square with a horizontal white line going down the middle. Neurons using the vertical filter would only look at central vertical lines as everything else will get multiplied by 0 and therefore blacked out. The same applies to the horizontal filter. In real life examples the filters are a bit more complicated than the above but the same principle applies. Additionally in real examples convolutional layers have many filters that will output one feature map per filter so it will actually result in a 3D representation of the input image. Lastly regarding the strides, sometimes the amount of pixels will mean that the receptive field will have some missing pixels. To combat this you can set padding to either fill the missing pixels to 0s or to ignore the receptive fields that don’t have all of the pixels available.

The other main difference to a normal neural network is the pooling layer. This layer’s role is to shrink the original image down into a smaller version. This is done to reduce computational load and the number of parameters that need to be tuned. Just like the convolutional layers, a neutron in a pooling layer is connected to a portion of the neutrons in the previous layer, within a smaller rectangle. However a pooling neuron does not have any weights, it’s only function is to aggregate its inputs using a aggregation function such as the mean or the max. Apart from reducing computational strain, pooling layers also provide a limited level of invariance which can be useful in classification tasks.

## Obtaining Spectrograms and Latent Features

As mentioned above, once we obtain the spotify id for a song we can get access to the preview of the track. We then turn this mp3 preview into a spectrogram using Pydub and Librosa. After creating enough labelled samples to form a dataset I then trained a neural network to identify the genre of the song using only the spectrogram. After training for a number of epochs, validation accuracy got to 0.6 which was satisfactory considering that some genres such as rap and grime are very similar. 

Following on from training the CNN, I removed the final layer and then got the model to predict on a dataset of songs that a couple of my friends had pooled together from their libraries and some official spotify playlists curated by DJs that my friends enjoy. I used this dataset as this process takes quite a lot of time so could not obtain a massive dataset. Therefore to increase the chances of song suggestions being well received, I used songs that I know are appealing to my audience. Additionally by sourcing these songs from more than one person there is enough variety in the data that some novel song recommendations will be made. Having said this, in the future I would like to increase the size of this data so more novel suggestions are made.

Now that we have a model and dataset whenever a user asks for a song that is unknown it can look for the spotify preview, turn the preview into a spectrogram, analyse the spectrogram and then find similar songs that way.

## Dashboard
 
        

