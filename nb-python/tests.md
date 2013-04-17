Testing
=======

Steps to get a reproducible common division of cases between training and 
testing (this should be better automated in the future):

1. Download the review corpus from http://www.cs.cornell.edu/people/pabo/movie-review-data/review_polarity.tar.gz
2. Copy prepcorpus.py to the directory which has the pos/ and neg/ directories
3. Run python prepcorpus.py
4. Two new directories will be created: postest/ and negtest/ for test cases. 
   The directories pos/ and neg/ will hold only training cases.

