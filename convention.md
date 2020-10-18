### List of conventions to follow to follow for debugging:

<ol>

<li> Model file should be named as model.pth </li>
<li> Data file should be named as data.csv (for loan prediction) </li>
<li> A model.pth should be present across all clients at time t = 0 </li>
<li> Each data file is unique to its bucket/container. A container has no access to any other container/bucket. However, for the purposes of this demo, they can share access to prevent from creating multiple service accounts. But the data transfer should be done as a 1:1 pair, and not looked up at any other container. </li>
<li> Model file should be named as model.pth </li>
<li> Containers on GCP and Azure all should have different and interpretable bucket/container names. No duplicates! </li>



</ol>
