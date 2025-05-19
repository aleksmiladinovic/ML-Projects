# AI Agent for Movie Recommendations

The files in this folder are used for running a local AI Agent for movie recommendations. The data used for the description of films is the same one as in the projects [Movie Recommendations](https://github.com/aleksmiladinovic/ML-Projects/tree/main/Movie%20Recommendations) and [Text Similarity of Movie Descriptions](https://github.com/aleksmiladinovic/TDA-Projects/tree/main/Text%20Similarity%20of%20Movie%20Descriptions).

The file ~agent.py~ contains the code for running our AI Agent, while the file ~vector.py~ contains the code for vectorizing the documents used by the model based on which answers are formed. Additionally, there is a variable ~limit~ which controls the number of entries that our local model considers, which is used purely for performance, as the users with weaker machines can consider smaller amount of data for faster responses.
