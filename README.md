# EECS 448 Final Project: BoomBox
Shrey Prakash Sahgal and Benjamin Steinig <br>
{shreyps, bsteinig}@umich.edu

## Abstract
We present a method for creating trajectory (temporal embedding) representations of songs and developing quantitative benchmarks for these representations on downstream MIR tasks. We explore related works in the field and explain how our system differs from previous implementations. We describe our music embeddings process and the web interface we developed to demonstrate our ability to perform music recommendation and genre classification using our novel representation. Finally, we present our results and discuss potential ethical concerns that come with developing a music embedding system.

## Examples

The BoomBox user interface allows users to select songs from an existing database or paste a YouTube link of a song. Users are encouraged to select at least 10 songs that so that the recommendation algorithm has more data to work with.
<img width="839" alt="BoomBox example 1" src="https://github.com/shreysahgal/eecs448-boombox/blob/76e7426febf054abcfbf61ba14474867efc18535/static/boombox_ui.png">

<br>
While generating the recommendations, BoomBox visualized trajectories of various songs that the user has chosen. The trajectories are very high dimensional -- each song is encoded into a $T \times 768$ vector, where $T$ is the number of timesteps. We use various dimension reduction algorithms to visualizes the song trajectories.
<img width="839" alt="BoomBox example 2" src="https://github.com/shreysahgal/eecs448-boombox/blob/76e7426febf054abcfbf61ba14474867efc18535/static/boombox_traj.png">
<br>
Finally, BoomBox generates song recommendations based on the user's input songs. Importantly, these recommendations are chosen using the learned encoding, which is based entirely off of the musical qualities of the songs.
<img width="839" alt="BoomBox example 3" src="https://github.com/shreysahgal/eecs448-boombox/blob/76e7426febf054abcfbf61ba14474867efc18535/static/boombox_recomendations.png">
