# -Music_Recommendation_System
Music recommendation system for DJs looking for new songs


I was speaking to one of my friends the other day about how DJs find songs that fit together. He told me that the way he and all his friends do it is by just listening to hours upon hours of tracks in the hopes that something good will turn up. This seemed like a very inefficient way of finding music so I set out to develop a system that would speed this process up. 
While developing this system I had a two main considerations. First, while we will be using matrix factorisation, because DJ's can sometimes want to use niche songs that basically no one has listened to we cannot totally rely on this method. Therefore there needs to be some form of content based recommendation engine. Second, as I want this system to be used by a non technical audience, I need to create an easy to use interface. These considerations will be dealt with in part 2 and 3 of this article trilogy while this article focuses on using Alternating Least Squares to recommend songs.
